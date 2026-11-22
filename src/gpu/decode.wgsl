// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Decoder kernels for one token: attention of one query per head over an F16 K/V cache, split over
// workgroups of keys, and F32 rows packed into such a cache. A cache of `cols` positions with d
// dims (all heads) holds f16 pairs: K transposed, word i·cols + j for pair i of position j, then
// V by position, word (d/2)·cols + j·d/2 + i. With K transposed, the threads of `attend` (one key
// each) read consecutive words.

struct Params {
  // attend: keys; pack: rows.
  n: u32,
  // attend: dims per head; pack: d.
  dim: u32,
  scale: f32,
  // The cache's positions.
  cols: u32,
  // pack: the position of the first row, and the first source float and floats per source row.
  col: u32,
  src: u32,
  stride: u32,
}

var<immediate> p: Params;

@group(0) @binding(0) var<storage, read> at_q: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read> at_kv: array<u32>;
@group(0) @binding(2) var<storage, read_write> at_part: array<vec2<f32>>;

// Keys per workgroup.
const KEYS: u32 = 256u;

// A key per thread, head_dim up to 128. `red` holds two 256-slot reduction regions so consecutive
// reductions in `attend` use distinct slots and need no trailing barrier.
var<workgroup> q: array<vec2<f32>, 64>;
var<workgroup> s: array<f32, 256>;
var<workgroup> red: array<f32, 512>;
var<workgroup> rv: array<vec2<f32>, 256>;

// Sum (or max) over the 256-thread workgroup in `red[off..]`. subgroup.wgsl has the fast version.
fn reduce_at(v: f32, lid: u32, is_max: bool, off: u32) -> f32 {
  red[off + lid] = v;
  workgroupBarrier();
  for (var h = 128u; h > 0u; h >>= 1u) {
    if (lid < h) {
      if (is_max) {
        red[off + lid] = max(red[off + lid], red[off + lid + h]);
      } else {
        red[off + lid] += red[off + lid + h];
      }
    }
    workgroupBarrier();
  }
  return red[off];
}

// Workgroup (c, h): head h's query against its keys c·KEYS.. (up to n): the scores' maximum m,
// the sum l of exp(score − m) and the sum of exp(score − m)·V, as part [h][c][pairs + 1]:
// the sums of V as pairs, then (m, l). `combine` merges the parts of each head. The query is
// rounded to f16, as the reference's flash attention stores it.
@compute @workgroup_size(256)
fn attend(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let hd = p.dim;
  let pairs = hd / 2u;
  let h = wg.y;
  let k0 = h * pairs * p.cols;
  let v_stride = num_workgroups.y * pairs;
  let v0 = v_stride * p.cols + h * pairs;
  if (lid < pairs) {
    let v = at_q[h * pairs + lid];
    q[lid] = vec2(to_half(v.x), to_half(v.y));
  }
  workgroupBarrier();
  let j0 = wg.x * KEYS;
  let n = min(p.n - j0, KEYS);
  var x = -3.0e38;
  if (lid < n) {
    var acc = 0.0;
    for (var i = 0u; i < pairs; i++) {
      let k = unpack2x16float(at_kv[k0 + i * p.cols + j0 + lid]);
      acc += dot(q[i], k);
    }
    x = acc * p.scale;
  }
  let m = reduce_at(x, lid, true, 0u);
  var e = 0.0;
  if (lid < n) {
    e = exp(x - m);
  }
  s[lid] = e;
  let l = reduce_at(e, lid, false, 256u);
  // Thread (g, c) sums V pair c over keys g, g + groups, ...
  let groups = 256u / pairs;
  let c = lid % pairs;
  let g = lid / pairs;
  var o = vec2(0.0);
  if (g < groups) {
    for (var j = g; j < n; j += groups) {
      o += s[j] * unpack2x16float(at_kv[v0 + (j0 + j) * v_stride + c]);
    }
  }
  rv[lid] = o;
  workgroupBarrier();
  let part = (h * ((p.n + KEYS - 1u) / KEYS) + wg.x) * (pairs + 1u);
  if (lid < pairs) {
    var t = vec2(0.0);
    for (var k = 0u; k < groups; k++) {
      t += rv[k * pairs + lid];
    }
    at_part[part + lid] = t;
  }
  if (lid == 0u) {
    at_part[part + pairs] = vec2(m, l);
  }
}

@group(0) @binding(0) var<storage, read> cb_part: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> cb_o: array<vec2<f32>>;

// One workgroup per head: attend's parts rescaled to their common maximum, summed and normalized.
@compute @workgroup_size(64)
fn combine(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let pairs = p.dim / 2u;
  if (lid >= pairs) {
    return;
  }
  let parts = (p.n + KEYS - 1u) / KEYS;
  let step = pairs + 1u;
  let base = wg.x * parts * step;
  if (parts == 1u) {
    cb_o[wg.x * pairs + lid] = cb_part[base + lid] / cb_part[base + pairs].y;
    return;
  }
  var m = -3.0e38;
  for (var c = 0u; c < parts; c++) {
    m = max(m, cb_part[base + c * step + pairs].x);
  }
  var l = 0.0;
  for (var c = 0u; c < parts; c++) {
    let ml = cb_part[base + c * step + pairs];
    l += exp(ml.x - m) * ml.y;
  }
  var o = vec2(0.0);
  for (var c = 0u; c < parts; c++) {
    let at = base + c * step;
    o += exp(cb_part[at + pairs].x - m) * cb_part[at + lid];
  }
  cb_o[wg.x * pairs + lid] = o / l;
}

@group(0) @binding(0) var<storage, read> pk_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> pk_y: array<u32>;

// n rows of x, each K then V (2·d floats from src + j·stride), into the cache at positions col..:
// pair i of row j is x[2i], x[2i + 1] (of V for i ≥ d/2). One row: a thread per pair; more:
// 32 rows × 32 pairs per workgroup (wg.z = 0 for K, 1 for V), so loads from pk_x and stores to
// both K (via a padded shared tile) and V are coalesced across consecutive lanes.
var<workgroup> tile: array<u32, 1056>;

@compute @workgroup_size(256)
fn pack(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let half = p.dim / 2u;
  if (p.n == 1u) {
    let i = wg.x * 256u + lid;
    if (i < p.dim) {
      let x = p.src + 2u * i;
      var at = i * p.cols + p.col;
      if (i >= half) {
        at = half * p.cols + p.col * half + i - half;
      }
      pk_y[at] = pack2x16float(vec2(to_half(pk_x[x]), to_half(pk_x[x + 1u])));
    }
  } else {
    let j0 = wg.x * 32u;
    let i0 = wg.y * 32u;
    let lx = lid % 32u;
    let ly = lid / 32u;
    if (wg.z != 0u) {
      let iv = i0 + lx;
      if (iv < half) {
        let base = p.src + p.dim + 2u * iv;
        let out_base = half * p.cols + p.col * half + iv;
        for (var k = 0u; k < 4u; k++) {
          let j = j0 + ly + k * 8u;
          if (j < p.n) {
            let x = base + j * p.stride;
            pk_y[out_base + j * half] =
              pack2x16float(vec2(to_half(pk_x[x]), to_half(pk_x[x + 1u])));
          }
        }
      }
    } else {
      let ik = i0 + lx;
      for (var k = 0u; k < 4u; k++) {
        let r = ly + k * 8u;
        let j = j0 + r;
        if (j < p.n && ik < half) {
          let x = p.src + j * p.stride + 2u * ik;
          tile[r * 33u + lx] = pack2x16float(vec2(to_half(pk_x[x]), to_half(pk_x[x + 1u])));
        }
      }
      workgroupBarrier();
      let jk = j0 + lx;
      if (jk < p.n) {
        for (var k = 0u; k < 4u; k++) {
          let r = ly + k * 8u;
          let i = i0 + r;
          if (i < half) {
            pk_y[i * p.cols + p.col + jk] = tile[lx * 33u + r];
          }
        }
      }
    }
  }
}

// gemm.wgsl's rounding to the nearest f16: pack2x16float truncates on DX12 (RTX 4060).
fn to_half(x: f32) -> f32 {
  let b = bitcast<u32>(x);
  let normal = bitcast<f32>((b + 0xfffu + ((b >> 13u) & 1u)) & 0xffffe000u);
  return select(normal, round(x * 0x1p24) * 0x1p-24, abs(x) < 0x1p-14);
}
