// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Subgroup versions of the decoder kernels' reductions (`gemm.wgsl`, `decode.wgsl`, `ops.wgsl`)
// and portable flash attention (`attention.wgsl`).

fn group_sums(s: vec4<f32>, lid: u32) -> vec4<f32> {
  return subgroupAdd(s);
}

fn reduce(v: f32, lid: u32, is_max: bool) -> f32 {
  let r = select(subgroupAdd(v), subgroupMax(v), is_max);
  if (lid % 32u == 0u) {
    red[lid / 32u] = r;
  }
  workgroupBarrier();
  var t = red[0];
  for (var i = 1u; i < 8u; i++) {
    if (is_max) {
      t = max(t, red[i]);
    } else {
      t += red[i];
    }
  }
  workgroupBarrier();
  return t;
}

fn reduce_at(v: f32, lid: u32, is_max: bool, off: u32) -> f32 {
  let r = select(subgroupAdd(v), subgroupMax(v), is_max);
  if (lid % 32u == 0u) {
    red[off + lid / 32u] = r;
  }
  workgroupBarrier();
  var t = red[off];
  for (var i = 1u; i < 8u; i++) {
    if (is_max) {
      t = max(t, red[off + i]);
    } else {
      t += red[off + i];
    }
  }
  return t;
}

fn max_pair(v: vec2<f32>, l: vec2<u32>) -> vec2<f32> {
  var t = max(v, subgroupShuffleXor(v, 1u));
  t = max(t, subgroupShuffleXor(t, 2u));
  return max(t, subgroupShuffleXor(t, 4u));
}

fn sum_pair(v: vec2<f32>, l: vec2<u32>) -> vec2<f32> {
  var s = v + subgroupShuffleXor(v, 1u);
  s += subgroupShuffleXor(s, 2u);
  return s + subgroupShuffleXor(s, 4u);
}

fn sync_kv() {
  workgroupBarrier();
}

fn probe_reduce(l: vec2<u32>, lid: u32, sg_size: u32) -> f32 {
  let chk = sum_pair(vec2(f32(l.y * 256u + (1u << l.x)), 0.0), l);
  let ok = sg_size >= 8u && chk.x == f32(8u * l.y * 256u + 255u);
  fp[lid] = vec2(select(0.0, 1.0, ok), 0.0);
  workgroupBarrier();
  var all_ok = 1.0;
  for (var i = 0u; i < 128u; i++) {
    all_ok *= fp[i].x;
  }
  return all_ok;
}
