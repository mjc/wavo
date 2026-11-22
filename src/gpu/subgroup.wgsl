// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
// Subgroup versions of reductions: `Portable::new` puts `reduce_sum`/`reduce_max` into ops.wgsl
// when `Features::SUBGROUP` is available (any subgroup width), and `Gpu::decoder` puts them (plus
// `group_sums` for 32-lane subgroups) into gemm.wgsl and decode.wgsl.

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
