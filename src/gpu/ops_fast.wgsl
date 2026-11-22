// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Fast LayerNorm (four rows per workgroup, one 32-lane subgroup per row). A dispatch with t == 0
// is a probe: it writes the subgroup size and count to the first two floats of the output.

struct Params {
  t: u32,
  ch: u32,
  head_dim: u32,
  scale: f32,
}

var<immediate> p: Params;

// LayerNorm, four rows per workgroup. With head_dim > 0 it writes [yr; y] into norm_yr.
// The rotary pass normalizes each element once.

@group(0) @binding(0) var<storage, read> norm_x: array<f32>;
@group(0) @binding(1) var<storage, read> norm_g: array<f32>;
@group(0) @binding(2) var<storage, read> norm_b: array<f32>;
@group(0) @binding(3) var<storage, read_write> norm_y: array<f32>;
@group(0) @binding(4) var<storage, read> rope: array<f32>;
@group(0) @binding(5) var<storage, read_write> norm_yr: array<f32>;

fn normalized(t: u32, i: u32, st: vec2<f32>) -> f32 {
  return (norm_x[t * p.ch + i] - st.x) * st.y * norm_g[i] + norm_b[i];
}

@compute @workgroup_size(128)
fn layer_norm(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_index) lane: u32,
  @builtin(subgroup_id) sg: u32,
  @builtin(subgroup_invocation_id) sl: u32,
  @builtin(subgroup_size) width: u32,
  @builtin(num_subgroups) count: u32,
) {
  if (p.t == 0u) {
    if (lane == 0u) {
      norm_y[0] = f32(width);
      norm_y[1] = f32(count);
    }
    return;
  }
  // Rows past the end compute on the last row and skip the stores.
  let r = wg.x * 4u + sg;
  let t = min(r, p.t - 1u);
  let ch4 = p.ch / 4u;
  var s = 0.0;
  for (var i = sl; i < ch4; i += 32u) {
    let o = t * p.ch + i * 4u;
    let v = vec4(norm_x[o], norm_x[o + 1u], norm_x[o + 2u], norm_x[o + 3u]);
    s += v.x + v.y + v.z + v.w;
  }
  for (var i = ch4 * 4u + sl; i < p.ch; i += 32u) {
    s += norm_x[t * p.ch + i];
  }
  let mean = subgroupAdd(s) / f32(p.ch);
  var q = 0.0;
  for (var i = sl; i < ch4; i += 32u) {
    let o = t * p.ch + i * 4u;
    let d = vec4(norm_x[o], norm_x[o + 1u], norm_x[o + 2u], norm_x[o + 3u]) - mean;
    q += dot(d, d);
  }
  for (var i = ch4 * 4u + sl; i < p.ch; i += 32u) {
    let d = norm_x[t * p.ch + i] - mean;
    q += d * d;
  }
  let st = vec2(mean, inverseSqrt(subgroupAdd(q) / f32(p.ch) + 1e-5));
  if (r >= p.t) {
    return;
  }
  if (p.head_dim == 0u) {
    for (var i = sl; i < ch4; i += 32u) {
      let c = i * 4u;
      let o = t * p.ch + c;
      let x = vec4(norm_x[o], norm_x[o + 1u], norm_x[o + 2u], norm_x[o + 3u]);
      let g = vec4(norm_g[c], norm_g[c + 1u], norm_g[c + 2u], norm_g[c + 3u]);
      let b = vec4(norm_b[c], norm_b[c + 1u], norm_b[c + 2u], norm_b[c + 3u]);
      let y = (x - st.x) * st.y * g + b;
      norm_y[o] = y.x;
      norm_y[o + 1u] = y.y;
      norm_y[o + 2u] = y.z;
      norm_y[o + 3u] = y.w;
    }
    for (var i = ch4 * 4u + sl; i < p.ch; i += 32u) {
      norm_y[t * p.ch + i] = normalized(t, i, st);
    }
    return;
  }
  let half = p.head_dim / 2u;
  for (var j = sl; j < p.ch / 2u; j += 32u) {
    let f = j % half;
    let i = j / half * p.head_dim + f;
    let v0 = normalized(t, i, st);
    let v1 = normalized(t, i + half, st);
    let cs = rope[t * p.head_dim + f];
    let sn = rope[t * p.head_dim + half + f];
    norm_yr[t * p.ch + i] = v0 * cs - v1 * sn;
    norm_yr[t * p.ch + i + half] = v1 * cs + v0 * sn;
    norm_yr[(p.t + t) * p.ch + i] = v0;
    norm_yr[(p.t + t) * p.ch + i + half] = v1;
  }
}
