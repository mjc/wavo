// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Linear layer c = epilogue(a·Wᵀ + bias): `a` is row-major [m][k] F32, W is [n][k] in its GGUF
// type. A workgroup computes a ROWS×COLS tile of c. K advances 32 per step (one block of 32). W
// has a multiple of 64 rows and k is a multiple of 32. A Q4 or Q5 conv whose stored row is
// shorter sets `src_k`; columns from there to `k` are zero. c's rows are n + dual
// wide, and outputs from n on (W's zero rows) are not written. W and `a` are rounded to f16, as
// ggml-metal's `kernel_mul_mm` (`ggml/src/ggml-metal/kernels/mul_mm.metal`) does. Without the
// rounding of W, near-ties (one CTC frame of e2e-ctc on ru-short) fall on the other side; without
// the rounding of `a`, the ivrit he-long near-tie does. The module also holds `gemv`, for one row
// of `a` (not rounded, as `mul_mv`), and `embed`.

// Weight type, one pipeline each: 0: F32, 1: F16, 2: Q8_0, 3: Q5_0, 4: Q4_0, 5: Q4_1, 6: Q5_1,
// 7: Q4_K, 8: Q6_K, 9: Q5_K.
override WTYPE: u32;
// Rows of the GEMM's tile: 64, or 32 (more workgroups).
override ROWS: u32 = 64u;
// Columns of the tile: 128, or 64 (one column group per thread). Q4_K passes 64.
override COLS: u32 = 128u;
// 32 k-steps by COLS/4 column quads.
override WT: u32 = 1024u;
// Subgroups of 32 threads in `gemv` (workgroup size 32 * GROUPS).
override GROUPS: u32 = 8u;
override RED: u32 = 128u * GROUPS;

struct Params {
  m: u32,
  // Output columns per row of c (before the dual ones).
  n: u32,
  k: u32,
  // With v = a·Wᵀ + bias, c = v, 1: SiLU(v), 2: ReLU(v), 3: c + alpha·v, 4: GELU(v), 5: c + GELU(v),
  // 6: tanh(v).
  mode: u32,
  alpha: f32,
  // Rows of c are n + dual wide: outputs j < dual are written again at n + j with bias[n + j].
  dual: u32,
  // gemv: c's first output; embed: the first float of the position's row in `a`.
  offset: u32,
  // Q4 and Q5 conv rows shorter than `k`: the file's blocks use this stride, and columns from
  // here to `k` are zero. 0 means the stored row is already `k` wide.
  src_k: u32,
  // With split > 0, columns from `split` on read `a` from row `m` on (`[a0; a1]`).
  split: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
// F32 as bits, F16 as pairs, Q8_0 as 4 int8 per word, Q4 as nibbles (16 bytes per 32), Q5 as
// 5-bit codes (24 bytes per 32), Q4_K as nibbles (128 bytes per 256), Q6_K as i8 codes already
// minus 32 (256 bytes per 256), Q5_K as bytes 0..31 (256 bytes per 256).
@group(0) @binding(3) var<storage, read> w: array<u32>;
// Q8_0, Q5_0 and Q4_0: one f16 scale per block, two per word. Q4_1 and Q5_1: an f16 scale and an
// f16 minimum, one pair per word. Q4_K: eight f32 pairs per 256 weights, `d·scale` and `dmin·min`
// for each group of 32, formed at upload. Q6_K: 20 bytes per 256 (f16 scale, sixteen i8 scales,
// two pad bytes). Q5_K uses a 20-byte record: f16 scale, f16 minimum, eight scales, eight
// minimums. Any buffer for the other types.
@group(0) @binding(4) var<storage, read> scales: array<u32>;
// `a` for 16-byte loads in `gemm` and `gemv`.
@group(0) @binding(0) var<storage, read> a4: array<vec4<f32>>;

// Rounds to the nearest f16, ties to even, by hand: NVIDIA's Vulkan driver folds
// unpack2x16float(pack2x16float(x)) into x and AMD's truncates. Normal halves keep 10 mantissa
// bits, subnormal ones are multiples of 2^-24. Values past the f16 range keep their exponent
// rather than become inf.
fn to_half(x: f32) -> f32 {
  let b = bitcast<u32>(x);
  let normal = bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
  return select(normal, round(x * 0x1p24) * 0x1p-24, abs(x) < 0x1p-14);
}

fn scale(block: u32) -> f32 {
  let s = unpack2x16float(scales[block / 2u]);
  return select(s.x, s.y, (block & 1u) == 1u);
}

// The u32 that holds four Q5 codes starting at weight x (a multiple of 4). Codes are packed
// little-endian, five bits each, 24 bytes per 32. gemm_fast.wgsl has the same load.
fn q5_word(x: u32) -> u32 {
  let block = x / 32u;
  let group = (x % 32u) / 4u;
  let bit = group * 20u;
  let byte = block * 24u + (bit >> 3u);
  let i = byte >> 2u;
  let s = (byte & 3u) * 8u;
  var word = w[i];
  if (s != 0u) {
    word = (word >> s) | (w[i + 1u] << (32u - s));
  }
  return word >> (bit & 7u);
}

fn q5_codes4(x: u32) -> vec4<f32> {
  let packed = q5_word(x);
  return vec4(f32(packed & 31u), f32((packed >> 5u) & 31u), f32((packed >> 10u) & 31u), f32((packed >> 15u) & 31u));
}

fn q5_4(x: u32) -> vec4<f32> {
  return (q5_codes4(x) - 16.0) * scale(x / 32u);
}

// Four Q4 codes at weight x (a multiple of 4). Nibbles are in weight order, 16 bytes per 32.
fn q4_codes4(x: u32) -> vec4<f32> {
  let block = x / 32u;
  let local = x % 32u;
  let word = w[block * 4u + local / 8u];
  let bits = (word >> ((local % 8u) * 4u)) & 0xFFFFu;
  return vec4(f32(bits & 15u), f32((bits >> 4u) & 15u), f32((bits >> 8u) & 15u), f32((bits >> 12u) & 15u));
}

fn q4_4(x: u32) -> vec4<f32> {
  return (q4_codes4(x) - 8.0) * scale(x / 32u);
}

// f16 scale and f16 minimum of one Q4_1 or Q5_1 block.
fn dm(block: u32) -> vec2<f32> {
  return unpack2x16float(scales[block]);
}

fn q41_4(x: u32) -> vec4<f32> {
  let s = dm(x / 32u);
  return q4_codes4(x) * s.x + s.y;
}

fn q51_4(x: u32) -> vec4<f32> {
  let s = dm(x / 32u);
  return q5_codes4(x) * s.x + s.y;
}

// Stride of a stored Q4 or Q5 row. Padded conv columns are not stored.
fn src_stride() -> u32 {
  return select(p.k, p.src_k, p.src_k != 0u);
}

fn q4_one(e: u32) -> f32 {
  return lane(q4_4(e & ~3u), e);
}

fn q5_one(e: u32) -> f32 {
  return lane(q5_4(e & ~3u), e);
}

fn q41_one(e: u32) -> f32 {
  return lane(q41_4(e & ~3u), e);
}

fn q51_one(e: u32) -> f32 {
  return lane(q51_4(e & ~3u), e);
}

// Four Q4 weights at `col` of a stored row. A row that already fills `k` is one aligned load.
fn q4_at(row: u32, col: u32) -> vec4<f32> {
  if (p.src_k == 0u) {
    return q4_4(row * p.k + col);
  }
  let src = src_stride();
  let e = row * src + col;
  return vec4<f32>(
    select(0.0, q4_one(e), col < src),
    select(0.0, q4_one(e + 1u), col + 1u < src),
    select(0.0, q4_one(e + 2u), col + 2u < src),
    select(0.0, q4_one(e + 3u), col + 3u < src),
  );
}

fn q5_at(row: u32, col: u32) -> vec4<f32> {
  if (p.src_k == 0u) {
    return q5_4(row * p.k + col);
  }
  let src = src_stride();
  let e = row * src + col;
  return vec4<f32>(
    select(0.0, q5_one(e), col < src),
    select(0.0, q5_one(e + 1u), col + 1u < src),
    select(0.0, q5_one(e + 2u), col + 2u < src),
    select(0.0, q5_one(e + 3u), col + 3u < src),
  );
}

fn q41_at(row: u32, col: u32) -> vec4<f32> {
  if (p.src_k == 0u) {
    return q41_4(row * p.k + col);
  }
  let src = src_stride();
  let e = row * src + col;
  return vec4<f32>(
    select(0.0, q41_one(e), col < src),
    select(0.0, q41_one(e + 1u), col + 1u < src),
    select(0.0, q41_one(e + 2u), col + 2u < src),
    select(0.0, q41_one(e + 3u), col + 3u < src),
  );
}

fn q51_at(row: u32, col: u32) -> vec4<f32> {
  if (p.src_k == 0u) {
    return q51_4(row * p.k + col);
  }
  let src = src_stride();
  let e = row * src + col;
  return vec4<f32>(
    select(0.0, q51_one(e), col < src),
    select(0.0, q51_one(e + 1u), col + 1u < src),
    select(0.0, q51_one(e + 2u), col + 2u < src),
    select(0.0, q51_one(e + 3u), col + 3u < src),
  );
}

// Four Q4_K codes at weight x (a multiple of 4). Nibbles are in weight order, eight per word.
fn q4k_codes4(x: u32) -> vec4<f32> {
  let bits = (w[x >> 3u] >> ((x & 4u) << 2u)) & 0xFFFFu;
  return vec4(f32(bits & 15u), f32((bits >> 4u) & 15u), f32((bits >> 8u) & 15u), f32((bits >> 12u) & 15u));
}

// `d · scale` and `dmin · min` for the group of 32 that contains x. Upload stored the product.
fn q4k_dm(x: u32) -> vec2<f32> {
  let i = (x >> 5u) << 1u;
  return vec2(bitcast<f32>(scales[i]), bitcast<f32>(scales[i + 1u]));
}

fn q4k_4(x: u32) -> vec4<f32> {
  let s = q4k_dm(x);
  return q4k_codes4(x) * s.x - s.y;
}

// Four Q6_K weights at x (a multiple of 4). Codes are i8 in weight order, already −32. Sixteen
// weights share one i8 scale, so the four codes share it. The scale record is f16 d, then the
// scales, then two pad bytes.
fn q6k_4(x: u32) -> vec4<f32> {
  let block = x / 256u;
  let local = x % 256u;
  let codes = vec4<f32>(unpack4xI8(w[block * 64u + local / 4u]));
  let base = block * 5u;
  let byte = 2u + local / 16u;
  let shift = (byte % 4u) * 8u;
  let raw = (scales[base + byte / 4u] >> shift) & 255u;
  let sc = select(f32(raw), f32(raw) - 256.0, raw >= 128u);
  let d = unpack2x16float(scales[base]).x;
  return codes * (d * sc);
}

// `d · scale` and `dmin · min` for the Q5_K group of 32 that contains x. The record is 20 bytes:
// f16 scale, f16 minimum, eight scales, eight minimums.
fn q5k_dm(x: u32) -> vec2<f32> {
  let block = x / 256u;
  let sub = (x % 256u) / 32u;
  let base = block * 5u;
  let hd = unpack2x16float(scales[base]);
  let shift = (sub % 4u) * 8u;
  let scale = (scales[base + 1u + sub / 4u] >> shift) & 255u;
  let min = (scales[base + 3u + sub / 4u] >> shift) & 255u;
  return vec2(hd.x * f32(scale), hd.y * f32(min));
}

// Four Q5_K codes at x (a multiple of 4). Bytes 0..31 in weight order, 256 per block.
fn q5k_4(x: u32) -> vec4<f32> {
  let block = x / 256u;
  let local = x % 256u;
  let codes = vec4<f32>(unpack4xI8(w[block * 64u + local / 4u]));
  let s = q5k_dm(x);
  return codes * s.x - s.y;
}

fn lane(v: vec4<f32>, x: u32) -> f32 {
  switch x & 3u {
    case 0u: { return v.x; }
    case 1u: { return v.y; }
    case 2u: { return v.z; }
    default: { return v.w; }
  }
}

// The [ROWS][32] tile of a as [row][k quad], rows padded to 9 quads against bank conflicts; the
// [32][COLS] tile of W as [k][column quad]. W is stored as the to_half values, two f16 pairs per
// column quad, and the inner loop unpacks them back to those f32 values.
var<workgroup> at: array<vec4<f32>, 576>;
var<workgroup> wt16: array<vec2<u32>, WT>;

// GELU with erf as ggml-metal computes it: `erf_approx` (Abramowitz–Stegun 7.1.26) in
// ggml/src/ggml-metal/kernels/common.h, `0.5·x·(1 + erf(x/√2))` in unary.metal.
fn gelu(x: f32) -> f32 {
  let z = abs(0.70710678118654752440 * x);
  let t = 1.0 / (1.0 + 0.3275911 * z);
  let e = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-z * z);
  return 0.5 * x * (1.0 + sign(x) * e);
}

fn store_one(i: u32, v: f32) {
  switch p.mode {
    case 1u: { c[i] = v / (1.0 + exp(-v)); }
    case 2u: { c[i] = max(v, 0.0); }
    case 3u: { c[i] += p.alpha * v; }
    case 4u: { c[i] = gelu(v); }
    case 5u: { c[i] += gelu(v); }
    // tanh(±10) is ±1 in f32; some drivers' tanh overflows to NaN on large inputs.
    case 6u: { c[i] = tanh(clamp(v, -10.0, 10.0)); }
    default: { c[i] = v; }
  }
}

fn epilogue(m: u32, n: u32, acc: f32) {
  if (n >= p.n) {
    return;
  }
  let i = m * (p.n + p.dual) + n;
  store_one(i, acc + bias[n]);
  if (n < p.dual) {
    c[i + p.n] = acc + bias[p.n + n];
  }
}

fn store(m: u32, n0: u32, acc: vec4<f32>) {
  if (m < p.m) {
    epilogue(m, n0, acc.x);
    epilogue(m, n0 + 1u, acc.y);
    epilogue(m, n0 + 2u, acc.z);
    epilogue(m, n0 + 3u, acc.w);
  }
}

// The words of W[n][k..k + 4]; for Q8_0 one, the word of its block's scale and the block's
// parity. Q4 and Q5 return the four dequantized weights as bits. Zeros from row p.n on.
fn w_words(n: u32, k: u32) -> vec4<u32> {
  if (n >= p.n) {
    return vec4(0u);
  }
  let x = n * p.k + k;
  switch WTYPE {
    case 0u: { return vec4(w[x], w[x + 1u], w[x + 2u], w[x + 3u]); }
    case 1u: { return vec4(w[x / 2u], w[x / 2u + 1u], 0u, 0u); }
    case 3u: { return bitcast<vec4<u32>>(q5_at(n, k)); }
    case 4u: { return bitcast<vec4<u32>>(q4_at(n, k)); }
    case 5u: { return bitcast<vec4<u32>>(q41_at(n, k)); }
    case 6u: { return bitcast<vec4<u32>>(q51_at(n, k)); }
    case 7u: { return bitcast<vec4<u32>>(q4k_4(x)); }
    case 8u: { return bitcast<vec4<u32>>(q6k_4(x)); }
    case 9u: { return bitcast<vec4<u32>>(q5k_4(x)); }
    default: { return vec4(w[x / 4u], scales[x / 64u], (x / 32u) & 1u, 0u); }
  }
}

// The 4 weights of `w_words`, rounded.
fn w_floats(v: vec4<u32>) -> vec4<f32> {
  var x: vec4<f32>;
  switch WTYPE {
    case 0u: { x = bitcast<vec4<f32>>(v); }
    case 1u: { x = vec4(unpack2x16float(v.x), unpack2x16float(v.y)); }
    case 3u, 4u, 5u, 6u, 7u, 8u, 9u: { x = bitcast<vec4<f32>>(v); }
    default: {
      let s = unpack2x16float(v.y);
      x = vec4<f32>(unpack4xI8(v.x)) * select(s.x, s.y, v.z == 1u);
    }
  }
  return vec4(to_half(x.x), to_half(x.y), to_half(x.z), to_half(x.w));
}

fn half4(v: vec4<f32>) -> vec4<f32> {
  return vec4(to_half(v.x), to_half(v.y), to_half(v.z), to_half(v.w));
}

// What a thread stages per K step: a quads of rows r and r + 32, and W words of 4 rows.
struct Fetch {
  a0: vec4<f32>,
  a1: vec4<f32>,
  w0: vec4<u32>,
  w1: vec4<u32>,
  w2: vec4<u32>,
  w3: vec4<u32>,
}

// Thread lid's share of the K step from k0: a rows m0 + m_off + lid / 8 (and + 32) at quad
// lid % 8, W rows from n0 + 4·(lid / 8) at the quad lid % 8.
fn fetch(m0: u32, n0: u32, k0: u32, lid: u32, m_off: u32) -> Fetch {
  var f: Fetch;
  let r = m0 + lid / 8u;
  let k = k0 + 4u * (lid % 8u);
  if (r < p.m) {
    f.a0 = half4(a4[((r + m_off) * p.k + k) / 4u]);
  }
  if (ROWS == 64u && r + 32u < p.m) {
    f.a1 = half4(a4[((r + 32u + m_off) * p.k + k) / 4u]);
  }
  // Column quad lid/8. A 64-column tile has 16 quads; the other threads only load a.
  if (lid / 8u < COLS / 4u) {
    let n = n0 + 4u * (lid / 8u);
    f.w0 = w_words(n, k);
    f.w1 = w_words(n + 1u, k);
    f.w2 = w_words(n + 2u, k);
    f.w3 = w_words(n + 3u, k);
  }
  return f;
}

// A workgroup computes a ROWS×COLS tile of c. Thread (tx, ty) covers rows ty + 16i and columns
// 4tx.., four of them, and for COLS == 128 also 64 + 4tx.., summed over k in order. The next K
// step's loads are issued before this one's products, which hides their latency on GPUs with few
// workgroups. 64 consecutive threads share 8 column quads. The W tile holds those weights as f16 pairs.
@compute @workgroup_size(256)
fn gemm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let n0 = wg.x * COLS;
  let m0 = wg.y * ROWS;
  let m_off = select(0u, p.m, p.split > 0u && n0 >= p.split);
  let lane = lid % 64u;
  let tx = lane % 8u + 8u * ((lid / 64u) % 2u);
  let ty = lane / 8u + 8u * (lid / 128u);
  var c00: vec4<f32>;
  var c01: vec4<f32>;
  var c10: vec4<f32>;
  var c11: vec4<f32>;
  var c20: vec4<f32>;
  var c21: vec4<f32>;
  var c30: vec4<f32>;
  var c31: vec4<f32>;
  let r = lid / 8u;
  let kq = 4u * (lid % 8u);
  var f = fetch(m0, n0, 0u, lid, m_off);
  for (var k0 = 0u; k0 < p.k; k0 += 32u) {
    at[r * 9u + kq / 4u] = f.a0;
    if (ROWS == 64u) {
      at[(r + 32u) * 9u + kq / 4u] = f.a1;
    }
    // W transposed to [k][column quad].
    let stride = COLS / 4u;
    if (r < stride) {
      let x0 = w_floats(f.w0);
      let x1 = w_floats(f.w1);
      let x2 = w_floats(f.w2);
      let x3 = w_floats(f.w3);
      wt16[kq * stride + r] = vec2(pack2x16float(vec2(x0.x, x1.x)), pack2x16float(vec2(x2.x, x3.x)));
      wt16[(kq + 1u) * stride + r] = vec2(pack2x16float(vec2(x0.y, x1.y)), pack2x16float(vec2(x2.y, x3.y)));
      wt16[(kq + 2u) * stride + r] = vec2(pack2x16float(vec2(x0.z, x1.z)), pack2x16float(vec2(x2.z, x3.z)));
      wt16[(kq + 3u) * stride + r] = vec2(pack2x16float(vec2(x0.w, x1.w)), pack2x16float(vec2(x2.w, x3.w)));
    }
    if (k0 + 32u < p.k) {
      f = fetch(m0, n0, k0 + 32u, lid, m_off);
    }
    workgroupBarrier();
    for (var kq = 0u; kq < 8u; kq++) {
      let a0 = at[ty * 9u + kq];
      let a1 = at[(ty + 16u) * 9u + kq];
      var a2: vec4<f32>;
      var a3: vec4<f32>;
      if (ROWS == 64u) {
        a2 = at[(ty + 32u) * 9u + kq];
        a3 = at[(ty + 48u) * 9u + kq];
      }
      let stride = COLS / 4u;
      for (var j = 0u; j < 4u; j++) {
        let p0 = wt16[(4u * kq + j) * stride + tx];
        let b0 = vec4(unpack2x16float(p0.x), unpack2x16float(p0.y));
        c00 += a0[j] * b0;
        c10 += a1[j] * b0;
        if (ROWS == 64u) {
          c20 += a2[j] * b0;
          c30 += a3[j] * b0;
        }
        if (COLS == 128u) {
          let p1 = wt16[(4u * kq + j) * stride + 16u + tx];
          let b1 = vec4(unpack2x16float(p1.x), unpack2x16float(p1.y));
          c01 += a0[j] * b1;
          c11 += a1[j] * b1;
          if (ROWS == 64u) {
            c21 += a2[j] * b1;
            c31 += a3[j] * b1;
          }
        }
      }
    }
    workgroupBarrier();
  }
  let n = n0 + 4u * tx;
  store(m0 + ty, n, c00);
  store(m0 + ty + 16u, n, c10);
  if (COLS == 128u) {
    store(m0 + ty, n + 64u, c01);
    store(m0 + ty + 16u, n + 64u, c11);
  }
  if (ROWS == 64u) {
    store(m0 + ty + 32u, n, c20);
    store(m0 + ty + 48u, n, c30);
    if (COLS == 128u) {
      store(m0 + ty + 32u, n + 64u, c21);
      store(m0 + ty + 48u, n + 64u, c31);
    }
  }
}

// a[8j..8j + 8]·W[row][8j..8j + 8]. Q8_0 scales the int8 sum once per 8 weights, as ggml's
// `mul_mv` does; Q4 and Q5 dequantize each code. W is not rounded.
fn dot8(row: u32, j: u32, a0: vec4<f32>, a1: vec4<f32>) -> f32 {
  let x = row * p.k + 8u * j;
  switch WTYPE {
    case 0u: {
      let w0 = vec4(w[x], w[x + 1u], w[x + 2u], w[x + 3u]);
      let w1 = vec4(w[x + 4u], w[x + 5u], w[x + 6u], w[x + 7u]);
      return dot(bitcast<vec4<f32>>(w0), a0) + dot(bitcast<vec4<f32>>(w1), a1);
    }
    case 1u: {
      let h = x / 2u;
      let w0 = vec4(unpack2x16float(w[h]), unpack2x16float(w[h + 1u]));
      return dot(w0, a0) + dot(vec4(unpack2x16float(w[h + 2u]), unpack2x16float(w[h + 3u])), a1);
    }
    case 3u: { return dot(q5_at(row, 8u * j), a0) + dot(q5_at(row, 8u * j + 4u), a1); }
    case 4u: { return dot(q4_at(row, 8u * j), a0) + dot(q4_at(row, 8u * j + 4u), a1); }
    case 5u: { return dot(q41_at(row, 8u * j), a0) + dot(q41_at(row, 8u * j + 4u), a1); }
    case 6u: { return dot(q51_at(row, 8u * j), a0) + dot(q51_at(row, 8u * j + 4u), a1); }
    case 7u: { return dot(q4k_4(x), a0) + dot(q4k_4(x + 4u), a1); }
    case 8u: { return dot(q6k_4(x), a0) + dot(q6k_4(x + 4u), a1); }
    case 9u: {
      // x is a multiple of 8, so two words hold the eight codes and one group of 32 holds the scale.
      let block = x / 256u;
      let local = x % 256u;
      let c0 = vec4<f32>(unpack4xI8(w[block * 64u + local / 4u]));
      let c1 = vec4<f32>(unpack4xI8(w[block * 64u + local / 4u + 1u]));
      let s = q5k_dm(x);
      return dot(c0, a0) * s.x - (a0.x + a0.y + a0.z + a0.w) * s.y
        + dot(c1, a1) * s.x - (a1.x + a1.y + a1.z + a1.w) * s.y;
    }
    default: {
      let q = x / 4u;
      let d = dot(vec4<f32>(unpack4xI8(w[q])), a0) + dot(vec4<f32>(unpack4xI8(w[q + 1u])), a1);
      return d * scale(x / 32u);
    }
  }
}

var<workgroup> red: array<f32, RED>;

// The sums of s over each group of 32 threads (lid / 32), for all its threads. subgroup.wgsl has
// the fast version.
fn group_sums(s: vec4<f32>, lid: u32) -> vec4<f32> {
  let lane = lid % 32u;
  // [4][32] per group.
  let o = (lid - lane) * 4u + lane;
  red[o] = s.x;
  red[o + 32u] = s.y;
  red[o + 64u] = s.z;
  red[o + 96u] = s.w;
  workgroupBarrier();
  for (var h = 16u; h > 0u; h >>= 1u) {
    if (lane < h) {
      for (var r = 0u; r < 128u; r += 32u) {
        red[o + r] += red[o + r + h];
      }
    }
    workgroupBarrier();
  }
  return vec4(red[o - lane], red[o - lane + 32u], red[o - lane + 64u], red[o - lane + 96u]);
}

// c[offset + n] = epilogue(a·W[n] + bias[n]) for one row of a: GROUPS groups of 32 threads, 4
// rows of W per group, each thread over 8 weights at a time (8 contiguous words per 4 threads and
// row).
@compute @workgroup_size(32u * GROUPS)
fn gemv(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let lane = lid % 32u;
  let row = (wg.x * GROUPS + lid / 32u) * 4u;
  var s = vec4(0.0);
  for (var j = lane; j < p.k / 8u; j += 32u) {
    let a0 = a4[2u * j];
    let a1 = a4[2u * j + 1u];
    s += vec4(dot8(row, j, a0, a1), dot8(row + 1u, j, a0, a1), dot8(row + 2u, j, a0, a1), dot8(row + 3u, j, a0, a1));
  }
  let t = group_sums(s, lid);
  if (lane < 4u) {
    store_one(p.offset + row + lane, t[lane] + bias[row + lane]);
  }
}

@group(0) @binding(5) var<storage, read> ids: array<u32>;

// c = W[ids[m]] + a[offset..] + bias for k values: a row of an embedding table in its GGUF type,
// plus a position's row.
@compute @workgroup_size(256)
fn embed(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x;
  if (i >= p.k) {
    return;
  }
  let src = src_stride();
  let x = ids[p.m] * src + i;
  var v: f32;
  if (p.src_k != 0u && i >= p.src_k) {
    v = 0.0;
  } else {
    switch WTYPE {
      case 0u: { v = bitcast<f32>(w[x]); }
      case 1u: { v = unpack2x16float(w[x / 2u])[x % 2u]; }
      case 3u: { v = q5_one(x); }
      case 4u: { v = q4_one(x); }
      case 5u: { v = q41_one(x); }
      case 6u: { v = q51_one(x); }
      case 7u: { v = lane(q4k_4(x & ~3u), x); }
      case 8u: { v = lane(q6k_4(x & ~3u), x); }
      case 9u: { v = lane(q5k_4(x & ~3u), x); }
      default: { v = f32(unpack4xI8(w[x / 4u])[x % 4u]) * scale(x / 32u); }
    }
  }
  c[i] = v + a[p.offset + i] + bias[i];
}
