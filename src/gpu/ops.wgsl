// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Row-wise and convolution kernels over time-major [t][ch] F32 activations.

// Whether conv_glu runs LayerNorm (true) or a per-channel affine x·g + b (false).
override LAYER_NORM: bool = true;

// One immediate struct for all kernels: naga sizes a module's immediates from its first one.
struct Params {
  t: u32,
  ch: u32,
  head_dim: u32,
  scale: f32,
  // conv_glu: the depthwise kernel size; rotary: neox (1) or normal (0).
  kernel: u32,
  layer_norm: u32,
  // im2col: the conv's stride and padding (and `kernel`).
  stride: u32,
  pad: u32,
}

var<immediate> p: Params;

var<workgroup> red: array<f32, 256>;
// One row of up to 1024 channels.
var<workgroup> row: array<f32, 1024>;

// Sum or max over a 256-thread workgroup. subgroup.wgsl has the fast versions.
fn reduce_sum(v: f32, lid: u32) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      red[lid] += red[lid + s];
    }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

fn reduce_max(v: f32, lid: u32) -> f32 {
  red[lid] = v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      red[lid] = max(red[lid], red[lid + s]);
    }
    workgroupBarrier();
  }
  let r = red[0];
  workgroupBarrier();
  return r;
}

// LayerNorm (eps 1e-5) of row t, one workgroup per row. With head_dim > 0 it writes the
// rotated rows followed by the normalized rows [yr; y] into norm_yr: NEOX split-half rotation
// inside each head, using a [t][cos | sin] table of head_dim / 2 frequencies each.

@group(0) @binding(0) var<storage, read> norm_x: array<f32>;
@group(0) @binding(1) var<storage, read> norm_g: array<f32>;
@group(0) @binding(2) var<storage, read> norm_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> norm_y: array<f32>;
@group(0) @binding(4) var<storage, read> rope: array<f32>;
@group(0) @binding(5) var<storage, read_write> norm_yr: array<f32>;

@compute @workgroup_size(256)
fn layer_norm(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let t = wg.x;
  let base = t * p.ch;
  var s = 0.0;
  for (var i = lid; i < p.ch; i += 256u) {
    s += norm_x[base + i];
  }
  let mean = reduce_sum(s, lid) / f32(p.ch);
  var q = 0.0;
  for (var i = lid; i < p.ch; i += 256u) {
    let d = norm_x[base + i] - mean;
    q += d * d;
  }
  let rstd = inverseSqrt(reduce_sum(q, lid) / f32(p.ch) + 1e-5);
  for (var i = lid; i < p.ch; i += 256u) {
    let v = (norm_x[base + i] - mean) * rstd * norm_g[i] + norm_b[i];
    if (p.head_dim != 0u) {
      row[i] = v;
    } else {
      norm_y[base + i] = v;
    }
  }
  // Before the return: FXC (DX12) rejects a barrier after a return under a loop it deems varying.
  workgroupBarrier();
  if (p.head_dim == 0u) {
    return;
  }
  let half = p.head_dim / 2u;
  for (var i = lid; i < p.ch; i += 256u) {
    let d = i % p.head_dim;
    let f = d % half;
    let sn = rope[t * p.head_dim + half + f];
    var v = row[i] * rope[t * p.head_dim + f];
    if (d < half) {
      v -= row[i + half] * sn;
    } else {
      v += row[i - half] * sn;
    }
    norm_yr[base + i] = v;
    norm_yr[p.t * p.ch + base + i] = row[i];
  }
}

// im2col for a conv1d over x [t][ch]: col[i][c · kernel + k] = x[stride · i + k − pad][c], matching
// a ggml conv weight [kernel, ch, out]. Rows are padded with zeros to a multiple of 32 columns, as
// `Gpu::conv` pads the weight (80 mels: 240 → 256).

@group(0) @binding(0) var<storage, read> col_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> col_y: array<f32>;

@compute @workgroup_size(256)
fn im2col(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let i = wg.y;
  let e = wg.x * 256u + lid;
  let width = (p.ch * p.kernel + 31u) / 32u * 32u;
  if (e >= width) {
    return;
  }
  // Source frame plus the padding.
  let s = p.stride * i + e % p.kernel;
  var v = 0.0;
  if (e < p.ch * p.kernel && s >= p.pad && s - p.pad < p.t) {
    v = col_x[(s - p.pad) * p.ch + e / p.kernel];
  }
  col_y[i * width + e] = v;
}

// GLU over h [t][2·ch] into y [t][ch]: y[t][c] = h[t][c] / (1 + exp(-h[t][ch + c])).

@group(0) @binding(0) var<storage, read> glu_h: array<f32>;
@group(0) @binding(1) var<storage, read_write> glu_y: array<f32>;

@compute @workgroup_size(256)
fn glu(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let i = wg.x * 256u + lid;
  let c4 = p.ch / 4u;
  if (i < c4) {
    let c = i * 4u;
    let b0 = wg.y * 2u * p.ch + c;
    let b1 = b0 + p.ch;
    let x = vec4(glu_h[b0], glu_h[b0 + 1u], glu_h[b0 + 2u], glu_h[b0 + 3u]);
    let g = vec4(glu_h[b1], glu_h[b1 + 1u], glu_h[b1 + 2u], glu_h[b1 + 3u]);
    let y = x / (vec4(1.0) + exp(-g));
    let o = wg.y * p.ch + c;
    glu_y[o] = y.x;
    glu_y[o + 1u] = y.y;
    glu_y[o + 2u] = y.z;
    glu_y[o + 3u] = y.w;
  }
  if (wg.x == 0u) {
    let c = c4 * 4u + lid;
    if (c < p.ch) {
      let base = wg.y * 2u * p.ch + c;
      glu_y[wg.y * p.ch + c] = glu_h[base] / (1.0 + exp(-glu_h[base + p.ch]));
    }
  }
}

// Conformer conv module between the pointwise convs, one workgroup per frame: depthwise conv
// over GLU output h [t][ch] (odd kernel, centred, weight [kernel][ch]) plus bias, LayerNorm over
// channels or a per-channel affine, SiLU.

@group(0) @binding(0) var<storage, read> conv_h: array<f32>;
@group(0) @binding(1) var<storage, read> conv_w: array<f32>;
@group(0) @binding(2) var<storage, read> conv_b: array<f32>;
@group(0) @binding(3) var<storage, read> conv_g: array<f32>;
@group(0) @binding(4) var<storage, read> conv_beta: array<f32>;
@group(0) @binding(5) var<storage, read_write> conv_y: array<f32>;

@compute @workgroup_size(256)
fn conv_glu(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let t = wg.x;
  let pad = p.kernel / 2u;
  var s = 0.0;
  for (var i = lid; i < p.ch; i += 256u) {
    var acc = conv_b[i];
    for (var k = 0u; k < p.kernel; k++) {
      // Source frame plus the padding.
      let src = t + k;
      if (src >= pad && src - pad < p.t) {
        acc += conv_w[k * p.ch + i] * conv_h[(src - pad) * p.ch + i];
      }
    }
    if (LAYER_NORM) {
      row[i] = acc;
      s += acc;
    } else {
      let y = acc * conv_g[i] + conv_beta[i];
      conv_y[t * p.ch + i] = y / (1.0 + exp(-y));
    }
  }
  if (LAYER_NORM) {
    let mean = reduce_sum(s, lid) / f32(p.ch);
    var q = 0.0;
    for (var i = lid; i < p.ch; i += 256u) {
      let d = row[i] - mean;
      q += d * d;
    }
    let rstd = inverseSqrt(reduce_sum(q, lid) / f32(p.ch) + 1e-5);
    for (var i = lid; i < p.ch; i += 256u) {
      let y = (row[i] - mean) * rstd * conv_g[i] + conv_beta[i];
      conv_y[t * p.ch + i] = y / (1.0 + exp(-y));
    }
  }
}

// Softmax of each row of s [heads][t][t] in place (see attention.wgsl).

@group(0) @binding(0) var<storage, read_write> sm_s: array<f32>;

// One workgroup per row (query wg.x, head wg.y).
@compute @workgroup_size(256)
fn softmax(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let base = (wg.y * p.t + wg.x) * p.t;
  var mx = -3.0e38;
  for (var j = lid; j < p.t; j += 256u) {
    mx = max(mx, sm_s[base + j]);
  }
  mx = reduce_max(mx, lid);
  var sum = 0.0;
  for (var j = lid; j < p.t; j += 256u) {
    let e = exp(sm_s[base + j] - mx);
    sm_s[base + j] = e;
    sum += e;
  }
  let inv = 1.0 / reduce_sum(sum, lid);
  for (var j = lid; j < p.t; j += 256u) {
    sm_s[base + j] *= inv;
  }
}

// Rotary embedding in place over rows x [t][ch]: in the first p.stride heads, for 2i < p.kernel,
// the pair (2i, 2i + 1) (GPT-J, `GGML_ROPE_TYPE_NORMAL`), or with p.layer_norm = 1 the pair
// (i, i + p.kernel / 2) (GPT-NeoX), rotates by row r's position p.pad + r, from a table of
// p.kernel / 2 cosines, then sines, per position.

@group(0) @binding(0) var<storage, read> rp_table: array<f32>;
@group(0) @binding(1) var<storage, read_write> rp_x: array<f32>;

@compute @workgroup_size(256)
fn rotary(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
  let half = p.kernel / 2u;
  let e = wg.x * 256u + lid;
  if (e >= p.stride * half) {
    return;
  }
  let i = e % half;
  let neox = p.layer_norm == 1u;
  let o = wg.y * p.ch + e / half * p.head_dim + select(2u * i, i, neox);
  let o2 = o + select(1u, half, neox);
  let row = (p.pad + wg.y) * p.kernel;
  let cs = vec2(rp_table[row + i], rp_table[row + half + i]);
  let x = vec2(rp_x[o], rp_x[o2]);
  rp_x[o] = x.x * cs.x - x.y * cs.y;
  rp_x[o2] = x.x * cs.y + x.y * cs.x;
}

// GroupNorm with one group, in place over x [t][ch], split across up to 64 workgroups:
// `group_norm_stats` writes per-workgroup sum and sum of squares into `part` [128]; `group_norm`
// reduces them, computes mean and variance = E[x²] − mean², and normalizes.

@group(0) @binding(0) var<storage, read_write> gn_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> gns_part: array<f32>;
@group(0) @binding(1) var<storage, read> gn_w: array<f32>;
@group(0) @binding(2) var<storage, read> gn_b: array<f32>;
@group(0) @binding(3) var<storage, read> gn_part: array<f32>;

@compute @workgroup_size(256)
fn group_norm_stats(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) num_wgs: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let n = p.t * p.ch;
  let n4 = n / 4u;
  let gid = wg.x * 256u + lid;
  var s = 0.0;
  var q = 0.0;
  for (var i = gid; i < n4; i += num_wgs.x * 256u) {
    let k = i * 4u;
    let v = vec4(gn_x[k], gn_x[k + 1u], gn_x[k + 2u], gn_x[k + 3u]);
    s += v.x + v.y + v.z + v.w;
    q += dot(v, v);
  }
  if (wg.x == 0u) {
    for (var i = n4 * 4u + lid; i < n; i += 256u) {
      let v = gn_x[i];
      s += v;
      q += v * v;
    }
  }
  s = reduce_sum(s, lid);
  q = reduce_sum(q, lid);
  if (lid == 0u) {
    gns_part[wg.x] = s;
    gns_part[64u + wg.x] = q;
  }
}

@compute @workgroup_size(256)
fn group_norm(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(num_workgroups) num_wgs: vec3<u32>,
  @builtin(local_invocation_index) lid: u32,
) {
  let s = reduce_sum(select(0.0, gn_part[lid], lid < num_wgs.x), lid);
  let q = reduce_sum(select(0.0, gn_part[64u + lid], lid < num_wgs.x), lid);
  let n = p.t * p.ch;
  let mean = s / f32(n);
  let rstd = inverseSqrt(max(q / f32(n) - mean * mean, 0.0) + p.scale);
  let gid = wg.x * 256u + lid;
  if (p.ch % 4u == 0u) {
    let n4 = n / 4u;
    for (var i = gid; i < n4; i += num_wgs.x * 256u) {
      let k = i * 4u;
      let c = k % p.ch;
      let x = vec4(gn_x[k], gn_x[k + 1u], gn_x[k + 2u], gn_x[k + 3u]);
      let w = vec4(gn_w[c], gn_w[c + 1u], gn_w[c + 2u], gn_w[c + 3u]);
      let b = vec4(gn_b[c], gn_b[c + 1u], gn_b[c + 2u], gn_b[c + 3u]);
      let y = (x - mean) * rstd * w + b;
      gn_x[k] = y.x;
      gn_x[k + 1u] = y.y;
      gn_x[k + 2u] = y.z;
      gn_x[k + 3u] = y.w;
    }
  } else {
    for (var i = gid; i < n; i += num_wgs.x * 256u) {
      let c = i % p.ch;
      gn_x[i] = (gn_x[i] - mean) * rstd * gn_w[c] + gn_b[c];
    }
  }
}

// SwiGLU in place over one row h [x | gate] of 2·ch: h[c] = SiLU(gate[c]) · x[c].

@group(0) @binding(0) var<storage, read_write> sw_h: array<f32>;

@compute @workgroup_size(256)
fn swiglu(@builtin(global_invocation_id) id: vec3<u32>) {
  let c = id.x;
  if (c < p.ch) {
    let g = sw_h[p.ch + c];
    sw_h[c] = g / (1.0 + exp(-g)) * sw_h[c];
  }
}

// ReLU in place over x[0..p.t].

@group(0) @binding(0) var<storage, read_write> rl_x: array<f32>;

@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < p.t) {
    rl_x[id.x] = max(rl_x[id.x], 0.0);
  }
}

// Multiply x[0..p.t] by p.scale, in place. Parakeet applies sqrt(d_model) after subsampling when
// the file sets xscaling.

@group(0) @binding(0) var<storage, read_write> sc_x: array<f32>;

@compute @workgroup_size(256)
fn scale(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x < p.t) {
    sc_x[id.x] = sc_x[id.x] * p.scale;
  }
}

// The index of the largest of x[0..p.t], the lowest one on a tie, as the bits of am_out[0]: the
// maximum, then the largest −i where it occurs (exact in f32 below 2^24).

@group(0) @binding(0) var<storage, read> am_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> am_out: array<u32>;

@compute @workgroup_size(256)
fn argmax(@builtin(local_invocation_index) lid: u32) {
  var mx = -3.0e38;
  var idx = 0u;
  for (var i = lid; i < p.t; i += 256u) {
    let v = am_x[i];
    if (v > mx) {
      mx = v;
      idx = i;
    }
  }
  red[lid] = mx;
  row[lid] = bitcast<f32>(idx);
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (lid < s) {
      let b = red[lid + s];
      let ib = bitcast<u32>(row[lid + s]);
      if (b > red[lid] || (b == red[lid] && ib < bitcast<u32>(row[lid]))) {
        red[lid] = b;
        row[lid] = bitcast<f32>(ib);
      }
    }
    workgroupBarrier();
  }
  if (lid == 0u) {
    am_out[0] = bitcast<u32>(row[0]);
  }
}

// Writes the call's number p.ch to slot p.t of mk_marks once mk_after (bound only to order this
// after its last writer) is written: proof that a submission ran to its end.

@group(0) @binding(0) var<storage, read> mk_after: array<f32>;
@group(0) @binding(1) var<storage, read_write> mk_marks: array<u32>;

@compute @workgroup_size(1)
fn mark() {
  mk_marks[p.t] = p.ch | (bitcast<u32>(mk_after[0]) & 0u);
}
