// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Subgroup versions of ops reductions (`reduce_sum`, `reduce_max`, any width), decode's
// `reduce_at`, gemv's `group_sums`, and portable flash attention (`max_pair`, `sum_pair`,
// `sync_kv`, `probe_reduce`).

fn group_sums(s: vec4<f32>, lid: u32) -> vec4<f32> {
  return subgroupAdd(s);
}

fn reduce_sum(v: f32, lid: u32) -> f32 {
  let width = subgroupAdd(1u);
  let r = subgroupAdd(v);
  if (width >= 256u) {
    return r;
  }
  let count = 256u / width;
  let sl = lid % width;
  if (sl == 0u) {
    red[lid / width] = r;
  }
  workgroupBarrier();
  var t = 0.0;
  if (count <= width) {
    let p = red[select(0u, sl, sl < count)];
    t = subgroupAdd(select(0.0, p, sl < count));
  } else {
    for (var i = 0u; i < count; i++) {
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

fn reduce_max(v: f32, lid: u32) -> f32 {
  let width = subgroupAdd(1u);
  let r = subgroupMax(v);
  if (width >= 256u) {
    return r;
  }
  let count = 256u / width;
  let sl = lid % width;
  if (sl == 0u) {
    red[lid / width] = r;
  }
  workgroupBarrier();
  var t: f32;
  if (count <= width) {
    t = subgroupMax(red[select(0u, sl, sl < count)]);
  } else {
    t = red[0];
    for (var i = 1u; i < count; i++) {
      t = max(t, red[i]);
    }
  }
  workgroupBarrier();
  return t;
}
