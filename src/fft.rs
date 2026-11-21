// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Real-input FFTs for the frontends' sizes (320, 400 and 512), written in plain autovectorizable
//! safe Rust with 8 frames packed across contiguous SIMD lanes (`[f32; 8]` / `[f64; 8]`):
//! - `Rfft8`: 8-frame batched `f32` real-to-complex FFT (320, 400) packing each frame into an `n /
//!   2`-point mixed-radix `{2, 4, 5}` complex FFT with the windowed gather fused into stage 0.
//! - `Rfft8F64`: 8-frame batched `f64` real-to-complex FFT for power-of-4 half-sizes (`512 -> 256`)
//!   with stage 0 fused into the windowed gather.

/// Stores eight `f32` power values into `dst` (as `f32` or widened to `f64`).
pub trait PowerOut: Copy + Default {
  fn store8(v: [f32; 8], dst: &mut [Self; 8]);
}

impl PowerOut for f32 {
  #[inline(always)]
  fn store8(v: [f32; 8], dst: &mut [f32; 8]) {
    *dst = v;
  }
}

impl PowerOut for f64 {
  #[inline(always)]
  fn store8(v: [f32; 8], dst: &mut [f64; 8]) {
    for (d, x) in dst.iter_mut().zip(v) {
      *d = x as f64;
    }
  }
}

enum Stage {
  Radix2 { stride: usize, tw: Vec<[f32; 2]> },
  Radix4 { stride: usize, tw: Vec<[f32; 6]> },
  Radix5 { stride: usize, tw: Vec<[f32; 8]> },
}

/// 8-frame batched `f32` real-to-complex FFT of even length `n` whose half `n / 2` factors into
/// 2 and 5 (covers GigaAM's 320 and Whisper's 400).
pub struct Rfft8 {
  n: usize,
  /// Radix of stage 0 (fused with the windowed gather).
  stage0: u8,
  /// `2 * perm[i]` for `i in 0..n/2`: the even sample index of digit-reversed pair `i`.
  perm2: Vec<u16>,
  /// In-place mixed-radix stages after stage 0.
  stages: Vec<Stage>,
  /// `(cos(2πk/n), sin(2πk/n))` for `k = 1..n/4`, used to unpack the real spectrum.
  unpack: Vec<[f32; 2]>,
}

impl Rfft8 {
  pub fn new(n: usize) -> Self {
    assert!(n >= 4 && n.is_multiple_of(2), "Rfft8 needs an even length >= 4, got {n}");
    let m = n / 2;
    let mut rem = m;
    let mut radices = Vec::new();
    while rem.is_multiple_of(5) {
      radices.push(5usize);
      rem /= 5;
    }
    while rem.is_multiple_of(4) {
      radices.push(4usize);
      rem /= 4;
    }
    while rem.is_multiple_of(2) {
      radices.push(2usize);
      rem /= 2;
    }
    assert_eq!(rem, 1, "Rfft8 length {n} has unsupported odd factor {rem}");

    let mut perm2 = vec![0u16; m];
    for (i, slot) in perm2.iter_mut().enumerate() {
      let (mut x, mut p) = (i, 0usize);
      for &r in &radices {
        p = p * r + x % r;
        x /= r;
      }
      *slot = (2 * p) as u16;
    }

    let trig = |num: usize, den: usize| -> (f32, f32) {
      let phase = std::f64::consts::TAU * num as f64 / den as f64;
      (phase.cos() as f32, phase.sin() as f32)
    };

    let stage0 = radices[0] as u8;
    let mut stages = Vec::with_capacity(radices.len().saturating_sub(1));
    let mut stride = radices[0];
    for &r in &radices[1..] {
      let group = r * stride;
      let stage = match r {
        2 => {
          let tw = (0..stride)
            .map(|u| {
              let (c, s) = trig(u, group);
              [c, s]
            })
            .collect();
          Stage::Radix2 { stride, tw }
        }
        4 => {
          let tw = (0..stride)
            .map(|u| {
              let (c1, s1) = trig(u, group);
              let (c2, s2) = trig(2 * u, group);
              let (c3, s3) = trig(3 * u, group);
              [c1, s1, c2, s2, c3, s3]
            })
            .collect();
          Stage::Radix4 { stride, tw }
        }
        5 => {
          let tw = (0..stride)
            .map(|u| {
              let (c1, s1) = trig(u, group);
              let (c2, s2) = trig(2 * u, group);
              let (c3, s3) = trig(3 * u, group);
              let (c4, s4) = trig(4 * u, group);
              [c1, s1, c2, s2, c3, s3, c4, s4]
            })
            .collect();
          Stage::Radix5 { stride, tw }
        }
        _ => unreachable!(),
      };
      stages.push(stage);
      stride = group;
    }

    let unpack = (1..m / 2)
      .map(|k| {
        let (c, s) = trig(k, n);
        [c, s]
      })
      .collect();
    Self { n, stage0, perm2, stages, unpack }
  }

  /// Packs eight `n`-sample frames multiplied by `window` (`n` long) directly through stage 0 into
  /// `re` and `im` (`n / 2` long each), then computes their power spectra `power` (`n / 2 + 1`).
  pub fn power_windowed<P: PowerOut>(
    &self,
    frames: [&[f32]; 8],
    window: &[f32],
    re: &mut [[f32; 8]],
    im: &mut [[f32; 8]],
    power: &mut [[P; 8]],
  ) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("fma") {
      // SAFETY: the CPU has FMA.
      unsafe { self.power_windowed_fma(frames, window, re, im, power) };
      return;
    }
    self.run_power_windowed(frames, window, re, im, power);
  }

  #[cfg(target_arch = "x86_64")]
  #[target_feature(enable = "fma")]
  fn power_windowed_fma<P: PowerOut>(
    &self,
    frames: [&[f32]; 8],
    window: &[f32],
    re: &mut [[f32; 8]],
    im: &mut [[f32; 8]],
    power: &mut [[P; 8]],
  ) {
    self.run_power_windowed(frames, window, re, im, power);
  }

  #[inline(always)]
  fn run_power_windowed<P: PowerOut>(
    &self,
    frames: [&[f32]; 8],
    window: &[f32],
    re: &mut [[f32; 8]],
    im: &mut [[f32; 8]],
    power: &mut [[P; 8]],
  ) {
    let n = self.n;
    let m = n / 2;
    let fs = [
      &frames[0][..n],
      &frames[1][..n],
      &frames[2][..n],
      &frames[3][..n],
      &frames[4][..n],
      &frames[5][..n],
      &frames[6][..n],
      &frames[7][..n],
    ];
    let w = &window[..n];
    let (re, im) = (&mut re[..m], &mut im[..m]);
    match self.stage0 {
      5 => load_stage_r5(fs, w, &self.perm2, re, im),
      4 => {
        gather_windowed_f32(fs, w, &self.perm2, re, im);
        stage0_r4(re, im);
      }
      2 => {
        gather_windowed_f32(fs, w, &self.perm2, re, im);
        stage0_r2(re, im);
      }
      _ => unreachable!(),
    }
    for stage in &self.stages {
      match stage {
        Stage::Radix2 { stride, tw } => stage_r2(re, im, *stride, tw),
        Stage::Radix4 { stride, tw } => stage_r4(re, im, *stride, tw),
        Stage::Radix5 { stride, tw } => stage_r5(re, im, *stride, tw),
      }
    }
    self.unpack_power(re, im, &mut power[..=m]);
  }

  #[inline(never)]
  fn unpack_power<P: PowerOut>(&self, re: &[[f32; 8]], im: &[[f32; 8]], power: &mut [[P; 8]]) {
    let m = self.n / 2;
    let half = m / 2;
    let (r0, i0) = (re[0], im[0]);
    let (mut dc, mut nyq) = ([0f32; 8], [0f32; 8]);
    for l in 0..8 {
      let a = r0[l] + i0[l];
      let b = r0[l] - i0[l];
      dc[l] = a * a;
      nyq[l] = b * b;
    }
    P::store8(dc, &mut power[0]);
    P::store8(nyq, &mut power[m]);
    if m.is_multiple_of(2) {
      let (rm, imv) = (re[half], im[half]);
      let mut pm = [0f32; 8];
      for l in 0..8 {
        pm[l] = rm[l].mul_add(rm[l], imv[l] * imv[l]);
      }
      P::store8(pm, &mut power[half]);
    }

    let (p_lo, p_hi) = power[1..m].split_at_mut(half);
    let p_lo = &mut p_lo[..half - 1];
    let hi_start = p_hi.len() - (half - 1);
    let p_hi = &mut p_hi[hi_start..];
    let re_lo = &re[1..half];
    let re_hi = &re[m - (half - 1)..m];
    let im_lo = &im[1..half];
    let im_hi = &im[m - (half - 1)..m];
    for ((((&ar, &ai), (&br, &bi)), (pk, pmk)), &[c, s]) in re_lo
      .iter()
      .zip(im_lo)
      .zip(re_hi.iter().rev().zip(im_hi.iter().rev()))
      .zip(p_lo.iter_mut().zip(p_hi.iter_mut().rev()))
      .zip(&self.unpack)
    {
      let (mut vk, mut vmk) = ([0f32; 8], [0f32; 8]);
      for l in 0..8 {
        let er = ar[l] + br[l];
        let ei = ai[l] - bi[l];
        let or = ai[l] + bi[l];
        let oi = br[l] - ar[l];
        let tr = oi.mul_add(s, or * c);
        let ti = (-or).mul_add(s, oi * c);
        let xkr = er + tr;
        let xki = ei + ti;
        let xmr = er - tr;
        let xmi = ti - ei;
        vk[l] = xkr.mul_add(xkr, xki * xki) * 0.25;
        vmk[l] = xmr.mul_add(xmr, xmi * xmi) * 0.25;
      }
      P::store8(vk, pk);
      P::store8(vmk, pmk);
    }
  }
}

#[inline(always)]
fn gather_pair8(fs: [&[f32]; 8], p: usize) -> ([f32; 8], [f32; 8]) {
  let (a0, a1) = (&fs[0][p..p + 2], &fs[1][p..p + 2]);
  let (a2, a3) = (&fs[2][p..p + 2], &fs[3][p..p + 2]);
  let (a4, a5) = (&fs[4][p..p + 2], &fs[5][p..p + 2]);
  let (a6, a7) = (&fs[6][p..p + 2], &fs[7][p..p + 2]);
  (
    [a0[0], a1[0], a2[0], a3[0], a4[0], a5[0], a6[0], a7[0]],
    [a0[1], a1[1], a2[1], a3[1], a4[1], a5[1], a6[1], a7[1]],
  )
}

#[inline(never)]
fn gather_windowed_f32(
  fs: [&[f32]; 8],
  w: &[f32],
  perm2: &[u16],
  re: &mut [[f32; 8]],
  im: &mut [[f32; 8]],
) {
  for ((dst_r, dst_i), &p) in re.iter_mut().zip(im.iter_mut()).zip(perm2) {
    let p = p as usize;
    let (gr, gi) = gather_pair8(fs, p);
    let (wr, wi) = (w[p], w[p + 1]);
    let (mut yr, mut yi) = ([0f32; 8], [0f32; 8]);
    for l in 0..8 {
      yr[l] = gr[l] * wr;
      yi[l] = gi[l] * wi;
    }
    *dst_r = yr;
    *dst_i = yi;
  }
}

#[inline(never)]
fn stage0_r2(re: &mut [[f32; 8]], im: &mut [[f32; 8]]) {
  let (mut cr, mut ci) = (re.chunks_exact_mut(4), im.chunks_exact_mut(4));
  for (gr, gi) in cr.by_ref().zip(ci.by_ref()) {
    let (z0r, z1r, z2r, z3r) = (gr[0], gr[1], gr[2], gr[3]);
    let (z0i, z1i, z2i, z3i) = (gi[0], gi[1], gi[2], gi[3]);
    let (mut y0r, mut y1r, mut y2r, mut y3r) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
    let (mut y0i, mut y1i, mut y2i, mut y3i) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
    for l in 0..8 {
      y0r[l] = z0r[l] + z1r[l];
      y1r[l] = z0r[l] - z1r[l];
      y0i[l] = z0i[l] + z1i[l];
      y1i[l] = z0i[l] - z1i[l];
      y2r[l] = z2r[l] + z3r[l];
      y3r[l] = z2r[l] - z3r[l];
      y2i[l] = z2i[l] + z3i[l];
      y3i[l] = z2i[l] - z3i[l];
    }
    gr[0] = y0r;
    gr[1] = y1r;
    gr[2] = y2r;
    gr[3] = y3r;
    gi[0] = y0i;
    gi[1] = y1i;
    gi[2] = y2i;
    gi[3] = y3i;
  }
  for (gr, gi) in
    cr.into_remainder().chunks_exact_mut(2).zip(ci.into_remainder().chunks_exact_mut(2))
  {
    let (z0r, z1r, z0i, z1i) = (gr[0], gr[1], gi[0], gi[1]);
    for l in 0..8 {
      gr[0][l] = z0r[l] + z1r[l];
      gr[1][l] = z0r[l] - z1r[l];
      gi[0][l] = z0i[l] + z1i[l];
      gi[1][l] = z0i[l] - z1i[l];
    }
  }
}

#[inline(never)]
fn stage0_r4(re: &mut [[f32; 8]], im: &mut [[f32; 8]]) {
  // Process two 4-point groups per inner loop so LLVM loop-vectorizes rather than full-unrolling.
  let (mut cr, mut ci) = (re.chunks_exact_mut(8), im.chunks_exact_mut(8));
  for (gr, gi) in cr.by_ref().zip(ci.by_ref()) {
    let (x0r, x1r, x2r, x3r) = (gr[0], gr[1], gr[2], gr[3]);
    let (x0i, x1i, x2i, x3i) = (gi[0], gi[1], gi[2], gi[3]);
    let (x4r, x5r, x6r, x7r) = (gr[4], gr[5], gr[6], gr[7]);
    let (x4i, x5i, x6i, x7i) = (gi[4], gi[5], gi[6], gi[7]);
    let (mut y0r, mut y1r, mut y2r, mut y3r) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
    let (mut y0i, mut y1i, mut y2i, mut y3i) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
    let (mut y4r, mut y5r, mut y6r, mut y7r) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
    let (mut y4i, mut y5i, mut y6i, mut y7i) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
    for l in 0..8 {
      let (a0r, a0i) = (x0r[l] + x2r[l], x0i[l] + x2i[l]);
      let (a1r, a1i) = (x0r[l] - x2r[l], x0i[l] - x2i[l]);
      let (b0r, b0i) = (x1r[l] + x3r[l], x1i[l] + x3i[l]);
      let (b1r, b1i) = (x1r[l] - x3r[l], x1i[l] - x3i[l]);
      y0r[l] = a0r + b0r;
      y0i[l] = a0i + b0i;
      y1r[l] = a1r + b1i;
      y1i[l] = a1i - b1r;
      y2r[l] = a0r - b0r;
      y2i[l] = a0i - b0i;
      y3r[l] = a1r - b1i;
      y3i[l] = a1i + b1r;

      let (c0r, c0i) = (x4r[l] + x6r[l], x4i[l] + x6i[l]);
      let (c1r, c1i) = (x4r[l] - x6r[l], x4i[l] - x6i[l]);
      let (d0r, d0i) = (x5r[l] + x7r[l], x5i[l] + x7i[l]);
      let (d1r, d1i) = (x5r[l] - x7r[l], x5i[l] - x7i[l]);
      y4r[l] = c0r + d0r;
      y4i[l] = c0i + d0i;
      y5r[l] = c1r + d1i;
      y5i[l] = c1i - d1r;
      y6r[l] = c0r - d0r;
      y6i[l] = c0i - d0i;
      y7r[l] = c1r - d1i;
      y7i[l] = c1i + d1r;
    }
    gr[0] = y0r;
    gr[1] = y1r;
    gr[2] = y2r;
    gr[3] = y3r;
    gr[4] = y4r;
    gr[5] = y5r;
    gr[6] = y6r;
    gr[7] = y7r;
    gi[0] = y0i;
    gi[1] = y1i;
    gi[2] = y2i;
    gi[3] = y3i;
    gi[4] = y4i;
    gi[5] = y5i;
    gi[6] = y6i;
    gi[7] = y7i;
  }
  for (gr, gi) in
    cr.into_remainder().chunks_exact_mut(4).zip(ci.into_remainder().chunks_exact_mut(4))
  {
    let (x0r, x1r, x2r, x3r) = (gr[0], gr[1], gr[2], gr[3]);
    let (x0i, x1i, x2i, x3i) = (gi[0], gi[1], gi[2], gi[3]);
    for l in 0..8 {
      let (a0r, a0i) = (x0r[l] + x2r[l], x0i[l] + x2i[l]);
      let (a1r, a1i) = (x0r[l] - x2r[l], x0i[l] - x2i[l]);
      let (b0r, b0i) = (x1r[l] + x3r[l], x1i[l] + x3i[l]);
      let (b1r, b1i) = (x1r[l] - x3r[l], x1i[l] - x3i[l]);
      gr[0][l] = a0r + b0r;
      gi[0][l] = a0i + b0i;
      gr[1][l] = a1r + b1i;
      gi[1][l] = a1i - b1r;
      gr[2][l] = a0r - b0r;
      gi[2][l] = a0i - b0i;
      gr[3][l] = a1r - b1i;
      gi[3][l] = a1i + b1r;
    }
  }
}

#[inline(never)]
fn load_stage_r5(
  fs: [&[f32]; 8],
  w: &[f32],
  perm2: &[u16],
  re: &mut [[f32; 8]],
  im: &mut [[f32; 8]],
) {
  const C1: f32 = 0.309_017;
  const S1: f32 = 0.951_056_54;
  const C2: f32 = -0.809_017;
  const S2: f32 = 0.587_785_24;
  for ((gr, gi), ps) in
    re.chunks_exact_mut(5).zip(im.chunks_exact_mut(5)).zip(perm2.chunks_exact(5))
  {
    let (p0, p1, p2, p3, p4) =
      (ps[0] as usize, ps[1] as usize, ps[2] as usize, ps[3] as usize, ps[4] as usize);
    let (g0r, g0i) = gather_pair8(fs, p0);
    let (g1r, g1i) = gather_pair8(fs, p1);
    let (g2r, g2i) = gather_pair8(fs, p2);
    let (g3r, g3i) = gather_pair8(fs, p3);
    let (g4r, g4i) = gather_pair8(fs, p4);
    let (w0r, w0i, w1r, w1i) = (w[p0], w[p0 + 1], w[p1], w[p1 + 1]);
    let (w2r, w2i, w3r, w3i, w4r, w4i) = (w[p2], w[p2 + 1], w[p3], w[p3 + 1], w[p4], w[p4 + 1]);
    let (mut y0r, mut y1r, mut y2r, mut y3r, mut y4r) =
      ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
    let (mut y0i, mut y1i, mut y2i, mut y3i, mut y4i) =
      ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
    for l in 0..8 {
      let (z0r, z0i) = (g0r[l] * w0r, g0i[l] * w0i);
      let (z1r, z1i) = (g1r[l] * w1r, g1i[l] * w1i);
      let (z2r, z2i) = (g2r[l] * w2r, g2i[l] * w2i);
      let (z3r, z3i) = (g3r[l] * w3r, g3i[l] * w3i);
      let (z4r, z4i) = (g4r[l] * w4r, g4i[l] * w4i);
      let (t1r, t1i) = (z1r + z4r, z1i + z4i);
      let (t2r, t2i) = (z2r + z3r, z2i + z3i);
      let (t3r, t3i) = (z1r - z4r, z1i - z4i);
      let (t4r, t4i) = (z2r - z3r, z2i - z3i);
      y0r[l] = z0r + (t1r + t2r);
      y0i[l] = z0i + (t1i + t2i);
      let m1r = t2r.mul_add(C2, t1r.mul_add(C1, z0r));
      let m1i = t2i.mul_add(C2, t1i.mul_add(C1, z0i));
      let m2r = t2r.mul_add(C1, t1r.mul_add(C2, z0r));
      let m2i = t2i.mul_add(C1, t1i.mul_add(C2, z0i));
      let m3r = t4i.mul_add(S2, t3i * S1);
      let m3i = t4r.mul_add(S2, t3r * S1);
      let m4r = (-t4i).mul_add(S1, t3i * S2);
      let m4i = (-t4r).mul_add(S1, t3r * S2);
      y1r[l] = m1r + m3r;
      y1i[l] = m1i - m3i;
      y2r[l] = m2r + m4r;
      y2i[l] = m2i - m4i;
      y3r[l] = m2r - m4r;
      y3i[l] = m2i + m4i;
      y4r[l] = m1r - m3r;
      y4i[l] = m1i + m3i;
    }
    gr[0] = y0r;
    gr[1] = y1r;
    gr[2] = y2r;
    gr[3] = y3r;
    gr[4] = y4r;
    gi[0] = y0i;
    gi[1] = y1i;
    gi[2] = y2i;
    gi[3] = y3i;
    gi[4] = y4i;
  }
}

#[inline(never)]
fn stage_r2(re: &mut [[f32; 8]], im: &mut [[f32; 8]], stride: usize, tw: &[[f32; 2]]) {
  let tw = &tw[..stride];
  for (gr, gi) in re.chunks_exact_mut(2 * stride).zip(im.chunks_exact_mut(2 * stride)) {
    let (r0, r1) = gr.split_at_mut(stride);
    let (i0, i1) = gi.split_at_mut(stride);
    for ((((a, b), c), d), &[wc, ws]) in
      r0.iter_mut().zip(r1.iter_mut()).zip(i0.iter_mut()).zip(i1.iter_mut()).zip(tw)
    {
      let (xa, xb, xc, xd) = (*a, *b, *c, *d);
      let (mut ya, mut yb, mut yc, mut yd) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
      for l in 0..8 {
        let tr = xd[l].mul_add(ws, xb[l] * wc);
        let ti = (-xb[l]).mul_add(ws, xd[l] * wc);
        ya[l] = xa[l] + tr;
        yb[l] = xa[l] - tr;
        yc[l] = xc[l] + ti;
        yd[l] = xc[l] - ti;
      }
      *a = ya;
      *b = yb;
      *c = yc;
      *d = yd;
    }
  }
}

#[inline(never)]
fn stage_r4(re: &mut [[f32; 8]], im: &mut [[f32; 8]], stride: usize, tw: &[[f32; 6]]) {
  let tw = &tw[..stride];
  for (gr, gi) in re.chunks_exact_mut(4 * stride).zip(im.chunks_exact_mut(4 * stride)) {
    let (r0, rest) = gr.split_at_mut(stride);
    let (r1, rest) = rest.split_at_mut(stride);
    let (r2, r3) = rest.split_at_mut(stride);
    let (i0, rest) = gi.split_at_mut(stride);
    let (i1, rest) = rest.split_at_mut(stride);
    let (i2, i3) = rest.split_at_mut(stride);
    for (u, &[c1, s1, c2, s2, c3, s3]) in tw.iter().enumerate() {
      let (x0r, x1r, x2r, x3r) = (r0[u], r1[u], r2[u], r3[u]);
      let (x0i, x1i, x2i, x3i) = (i0[u], i1[u], i2[u], i3[u]);
      let (mut y0r, mut y1r, mut y2r, mut y3r) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
      let (mut y0i, mut y1i, mut y2i, mut y3i) = ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
      for l in 0..8 {
        let (z0r, z0i) = (x0r[l], x0i[l]);
        let (z1r, z1i) = (x1i[l].mul_add(s1, x1r[l] * c1), (-x1r[l]).mul_add(s1, x1i[l] * c1));
        let (z2r, z2i) = (x2i[l].mul_add(s2, x2r[l] * c2), (-x2r[l]).mul_add(s2, x2i[l] * c2));
        let (z3r, z3i) = (x3i[l].mul_add(s3, x3r[l] * c3), (-x3r[l]).mul_add(s3, x3i[l] * c3));
        let (a0r, a0i) = (z0r + z2r, z0i + z2i);
        let (a1r, a1i) = (z0r - z2r, z0i - z2i);
        let (b0r, b0i) = (z1r + z3r, z1i + z3i);
        let (b1r, b1i) = (z1r - z3r, z1i - z3i);
        y0r[l] = a0r + b0r;
        y0i[l] = a0i + b0i;
        y1r[l] = a1r + b1i;
        y1i[l] = a1i - b1r;
        y2r[l] = a0r - b0r;
        y2i[l] = a0i - b0i;
        y3r[l] = a1r - b1i;
        y3i[l] = a1i + b1r;
      }
      r0[u] = y0r;
      r1[u] = y1r;
      r2[u] = y2r;
      r3[u] = y3r;
      i0[u] = y0i;
      i1[u] = y1i;
      i2[u] = y2i;
      i3[u] = y3i;
    }
  }
}

#[inline(never)]
fn stage_r5(re: &mut [[f32; 8]], im: &mut [[f32; 8]], stride: usize, tw: &[[f32; 8]]) {
  const C1: f32 = 0.309_017;
  const S1: f32 = 0.951_056_54;
  const C2: f32 = -0.809_017;
  const S2: f32 = 0.587_785_24;
  let tw = &tw[..stride];
  for (gr, gi) in re.chunks_exact_mut(5 * stride).zip(im.chunks_exact_mut(5 * stride)) {
    let (r0, rest) = gr.split_at_mut(stride);
    let (r1, rest) = rest.split_at_mut(stride);
    let (r2, rest) = rest.split_at_mut(stride);
    let (r3, r4) = rest.split_at_mut(stride);
    let (i0, rest) = gi.split_at_mut(stride);
    let (i1, rest) = rest.split_at_mut(stride);
    let (i2, rest) = rest.split_at_mut(stride);
    let (i3, i4) = rest.split_at_mut(stride);
    for (u, &[c1, s1, c2, s2, c3, s3, c4, s4]) in tw.iter().enumerate() {
      let (x0r, x1r, x2r, x3r, x4r) = (r0[u], r1[u], r2[u], r3[u], r4[u]);
      let (x0i, x1i, x2i, x3i, x4i) = (i0[u], i1[u], i2[u], i3[u], i4[u]);
      let (mut y0r, mut y1r, mut y2r, mut y3r, mut y4r) =
        ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
      let (mut y0i, mut y1i, mut y2i, mut y3i, mut y4i) =
        ([0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8], [0f32; 8]);
      for l in 0..8 {
        let (z0r, z0i) = (x0r[l], x0i[l]);
        let (z1r, z1i) = (x1i[l].mul_add(s1, x1r[l] * c1), (-x1r[l]).mul_add(s1, x1i[l] * c1));
        let (z2r, z2i) = (x2i[l].mul_add(s2, x2r[l] * c2), (-x2r[l]).mul_add(s2, x2i[l] * c2));
        let (z3r, z3i) = (x3i[l].mul_add(s3, x3r[l] * c3), (-x3r[l]).mul_add(s3, x3i[l] * c3));
        let (z4r, z4i) = (x4i[l].mul_add(s4, x4r[l] * c4), (-x4r[l]).mul_add(s4, x4i[l] * c4));
        let (t1r, t1i) = (z1r + z4r, z1i + z4i);
        let (t2r, t2i) = (z2r + z3r, z2i + z3i);
        let (t3r, t3i) = (z1r - z4r, z1i - z4i);
        let (t4r, t4i) = (z2r - z3r, z2i - z3i);
        y0r[l] = z0r + (t1r + t2r);
        y0i[l] = z0i + (t1i + t2i);
        let m1r = t2r.mul_add(C2, t1r.mul_add(C1, z0r));
        let m1i = t2i.mul_add(C2, t1i.mul_add(C1, z0i));
        let m2r = t2r.mul_add(C1, t1r.mul_add(C2, z0r));
        let m2i = t2i.mul_add(C1, t1i.mul_add(C2, z0i));
        let m3r = t4i.mul_add(S2, t3i * S1);
        let m3i = t4r.mul_add(S2, t3r * S1);
        let m4r = (-t4i).mul_add(S1, t3i * S2);
        let m4i = (-t4r).mul_add(S1, t3r * S2);
        y1r[l] = m1r + m3r;
        y1i[l] = m1i - m3i;
        y2r[l] = m2r + m4r;
        y2i[l] = m2i - m4i;
        y3r[l] = m2r - m4r;
        y3i[l] = m2i + m4i;
        y4r[l] = m1r - m3r;
        y4i[l] = m1i + m3i;
      }
      r0[u] = y0r;
      r1[u] = y1r;
      r2[u] = y2r;
      r3[u] = y3r;
      r4[u] = y4r;
      i0[u] = y0i;
      i1[u] = y1i;
      i2[u] = y2i;
      i3[u] = y3i;
      i4[u] = y4i;
    }
  }
}

/// 8-frame batched `f64` real-to-complex FFT for `n = 2 * 4^k` (Parakeet's `n = 512`, `m = 256`):
/// 8-lane `[f64; 8]` loops over non-aliasing locals autovectorize cleanly to NEON `float64x2_t`.
pub struct Rfft8F64 {
  n: usize,
  perm2: Vec<u16>,
  /// Radix-4 twiddles for stages `stride = 4, 16, 64, ...`.
  stages: Vec<(usize, Vec<[f64; 6]>)>,
  unpack: Vec<[f64; 2]>,
}

impl Rfft8F64 {
  pub fn new(n: usize) -> Self {
    let m = n / 2;
    let mut rem = m;
    let mut count = 0usize;
    while rem.is_multiple_of(4) {
      count += 1;
      rem /= 4;
    }
    assert!(n >= 8 && n.is_multiple_of(2) && rem == 1, "Rfft8F64 needs n = 2 * 4^k, got {n}");
    let mut perm2 = vec![0u16; m];
    for (i, slot) in perm2.iter_mut().enumerate() {
      let (mut x, mut p) = (i, 0usize);
      for _ in 0..count {
        p = (p << 2) | (x & 3);
        x >>= 2;
      }
      *slot = (2 * p) as u16;
    }
    let trig = |num: usize, den: usize| -> (f64, f64) {
      let phase = std::f64::consts::TAU * num as f64 / den as f64;
      (phase.cos(), phase.sin())
    };
    let mut stages = Vec::with_capacity(count - 1);
    let mut stride = 4usize;
    while stride < m {
      let group = 4 * stride;
      let tw = (0..stride)
        .map(|u| {
          let (c1, s1) = trig(u, group);
          let (c2, s2) = trig(2 * u, group);
          let (c3, s3) = trig(3 * u, group);
          [c1, s1, c2, s2, c3, s3]
        })
        .collect();
      stages.push((stride, tw));
      stride = group;
    }
    let unpack = (1..m / 2)
      .map(|k| {
        let (c, s) = trig(k, n);
        [c, s]
      })
      .collect();
    Self { n, perm2, stages, unpack }
  }

  /// Computes `f32` power spectra of eight `f64` frames into `power` (`n / 2 + 1` long).
  pub fn power_windowed_f32(
    &self,
    frames: [&[f64]; 8],
    window: &[f64],
    re: &mut [[f64; 8]],
    im: &mut [[f64; 8]],
    power: &mut [[f32; 8]],
  ) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("fma") {
      // SAFETY: the CPU has FMA.
      unsafe { self.power_windowed_f32_fma(frames, window, re, im, power) };
      return;
    }
    self.run_power_windowed_f32(frames, window, re, im, power);
  }

  #[cfg(target_arch = "x86_64")]
  #[target_feature(enable = "fma")]
  fn power_windowed_f32_fma(
    &self,
    frames: [&[f64]; 8],
    window: &[f64],
    re: &mut [[f64; 8]],
    im: &mut [[f64; 8]],
    power: &mut [[f32; 8]],
  ) {
    self.run_power_windowed_f32(frames, window, re, im, power);
  }

  #[inline(always)]
  fn run_power_windowed_f32(
    &self,
    frames: [&[f64]; 8],
    window: &[f64],
    re: &mut [[f64; 8]],
    im: &mut [[f64; 8]],
    power: &mut [[f32; 8]],
  ) {
    let n = self.n;
    let m = n / 2;
    let fs = [
      &frames[0][..n],
      &frames[1][..n],
      &frames[2][..n],
      &frames[3][..n],
      &frames[4][..n],
      &frames[5][..n],
      &frames[6][..n],
      &frames[7][..n],
    ];
    let w = &window[..n];
    let (re, im) = (&mut re[..m], &mut im[..m]);
    gather_windowed_f64(fs, w, &self.perm2, re, im);
    stage0_r4_f64(re, im);
    for (stride, tw) in &self.stages {
      stage_r4_f64(re, im, *stride, tw);
    }
    self.unpack_power_f32(re, im, &mut power[..=m]);
  }

  #[inline(never)]
  fn unpack_power_f32(&self, re: &[[f64; 8]], im: &[[f64; 8]], power: &mut [[f32; 8]]) {
    let m = self.n / 2;
    let half = m / 2;
    let (r0, i0) = (re[0], im[0]);
    let (rm, imv) = (re[half], im[half]);
    let (mut p0, mut pm, mut ph) = ([0f32; 8], [0f32; 8], [0f32; 8]);
    for l in 0..8 {
      let dc = r0[l] + i0[l];
      let nyq = r0[l] - i0[l];
      p0[l] = (dc * dc) as f32;
      pm[l] = (nyq * nyq) as f32;
      ph[l] = rm[l].mul_add(rm[l], imv[l] * imv[l]) as f32;
    }
    power[0] = p0;
    power[m] = pm;
    power[half] = ph;

    let (p_lo, p_hi) = power[1..m].split_at_mut(half);
    let p_lo = &mut p_lo[..half - 1];
    let hi_start = p_hi.len() - (half - 1);
    let p_hi = &mut p_hi[hi_start..];
    let re_lo = &re[1..half];
    let re_hi = &re[m - (half - 1)..m];
    let im_lo = &im[1..half];
    let im_hi = &im[m - (half - 1)..m];
    for ((((&ar, &ai), (&br, &bi)), (pk, pmk)), &[c, s]) in re_lo
      .iter()
      .zip(im_lo)
      .zip(re_hi.iter().rev().zip(im_hi.iter().rev()))
      .zip(p_lo.iter_mut().zip(p_hi.iter_mut().rev()))
      .zip(&self.unpack)
    {
      let (mut vk, mut vmk) = ([0f32; 8], [0f32; 8]);
      for l in 0..8 {
        let er = ar[l] + br[l];
        let ei = ai[l] - bi[l];
        let or = ai[l] + bi[l];
        let oi = br[l] - ar[l];
        let tr = oi.mul_add(s, or * c);
        let ti = (-or).mul_add(s, oi * c);
        let xkr = er + tr;
        let xki = ei + ti;
        let xmr = er - tr;
        let xmi = ti - ei;
        vk[l] = (xkr.mul_add(xkr, xki * xki) * 0.25) as f32;
        vmk[l] = (xmr.mul_add(xmr, xmi * xmi) * 0.25) as f32;
      }
      *pk = vk;
      *pmk = vmk;
    }
  }
}

#[inline(always)]
fn gather_pair8_f64(fs: [&[f64]; 8], p: usize) -> ([f64; 8], [f64; 8]) {
  let (a0, a1) = (&fs[0][p..p + 2], &fs[1][p..p + 2]);
  let (a2, a3) = (&fs[2][p..p + 2], &fs[3][p..p + 2]);
  let (a4, a5) = (&fs[4][p..p + 2], &fs[5][p..p + 2]);
  let (a6, a7) = (&fs[6][p..p + 2], &fs[7][p..p + 2]);
  (
    [a0[0], a1[0], a2[0], a3[0], a4[0], a5[0], a6[0], a7[0]],
    [a0[1], a1[1], a2[1], a3[1], a4[1], a5[1], a6[1], a7[1]],
  )
}

#[inline(never)]
fn gather_windowed_f64(
  fs: [&[f64]; 8],
  w: &[f64],
  perm2: &[u16],
  re: &mut [[f64; 8]],
  im: &mut [[f64; 8]],
) {
  for ((dst_r, dst_i), &p) in re.iter_mut().zip(im.iter_mut()).zip(perm2) {
    let p = p as usize;
    let (gr, gi) = gather_pair8_f64(fs, p);
    let (wr, wi) = (w[p], w[p + 1]);
    let (mut yr, mut yi) = ([0f64; 8], [0f64; 8]);
    for l in 0..8 {
      yr[l] = gr[l] * wr;
      yi[l] = gi[l] * wi;
    }
    *dst_r = yr;
    *dst_i = yi;
  }
}

#[inline(never)]
fn stage0_r4_f64(re: &mut [[f64; 8]], im: &mut [[f64; 8]]) {
  // Process two 4-point groups per inner loop so LLVM loop-vectorizes rather than full-unrolling.
  let (mut cr, mut ci) = (re.chunks_exact_mut(8), im.chunks_exact_mut(8));
  for (gr, gi) in cr.by_ref().zip(ci.by_ref()) {
    let (x0r, x1r, x2r, x3r) = (gr[0], gr[1], gr[2], gr[3]);
    let (x0i, x1i, x2i, x3i) = (gi[0], gi[1], gi[2], gi[3]);
    let (x4r, x5r, x6r, x7r) = (gr[4], gr[5], gr[6], gr[7]);
    let (x4i, x5i, x6i, x7i) = (gi[4], gi[5], gi[6], gi[7]);
    let (mut y0r, mut y1r, mut y2r, mut y3r) = ([0f64; 8], [0f64; 8], [0f64; 8], [0f64; 8]);
    let (mut y0i, mut y1i, mut y2i, mut y3i) = ([0f64; 8], [0f64; 8], [0f64; 8], [0f64; 8]);
    let (mut y4r, mut y5r, mut y6r, mut y7r) = ([0f64; 8], [0f64; 8], [0f64; 8], [0f64; 8]);
    let (mut y4i, mut y5i, mut y6i, mut y7i) = ([0f64; 8], [0f64; 8], [0f64; 8], [0f64; 8]);
    for l in 0..8 {
      let (a0r, a0i) = (x0r[l] + x2r[l], x0i[l] + x2i[l]);
      let (a1r, a1i) = (x0r[l] - x2r[l], x0i[l] - x2i[l]);
      let (b0r, b0i) = (x1r[l] + x3r[l], x1i[l] + x3i[l]);
      let (b1r, b1i) = (x1r[l] - x3r[l], x1i[l] - x3i[l]);
      y0r[l] = a0r + b0r;
      y0i[l] = a0i + b0i;
      y1r[l] = a1r + b1i;
      y1i[l] = a1i - b1r;
      y2r[l] = a0r - b0r;
      y2i[l] = a0i - b0i;
      y3r[l] = a1r - b1i;
      y3i[l] = a1i + b1r;

      let (c0r, c0i) = (x4r[l] + x6r[l], x4i[l] + x6i[l]);
      let (c1r, c1i) = (x4r[l] - x6r[l], x4i[l] - x6i[l]);
      let (d0r, d0i) = (x5r[l] + x7r[l], x5i[l] + x7i[l]);
      let (d1r, d1i) = (x5r[l] - x7r[l], x5i[l] - x7i[l]);
      y4r[l] = c0r + d0r;
      y4i[l] = c0i + d0i;
      y5r[l] = c1r + d1i;
      y5i[l] = c1i - d1r;
      y6r[l] = c0r - d0r;
      y6i[l] = c0i - d0i;
      y7r[l] = c1r - d1i;
      y7i[l] = c1i + d1r;
    }
    gr[0] = y0r;
    gr[1] = y1r;
    gr[2] = y2r;
    gr[3] = y3r;
    gr[4] = y4r;
    gr[5] = y5r;
    gr[6] = y6r;
    gr[7] = y7r;
    gi[0] = y0i;
    gi[1] = y1i;
    gi[2] = y2i;
    gi[3] = y3i;
    gi[4] = y4i;
    gi[5] = y5i;
    gi[6] = y6i;
    gi[7] = y7i;
  }
  for (gr, gi) in
    cr.into_remainder().chunks_exact_mut(4).zip(ci.into_remainder().chunks_exact_mut(4))
  {
    let (x0r, x1r, x2r, x3r) = (gr[0], gr[1], gr[2], gr[3]);
    let (x0i, x1i, x2i, x3i) = (gi[0], gi[1], gi[2], gi[3]);
    for l in 0..8 {
      let (a0r, a0i) = (x0r[l] + x2r[l], x0i[l] + x2i[l]);
      let (a1r, a1i) = (x0r[l] - x2r[l], x0i[l] - x2i[l]);
      let (b0r, b0i) = (x1r[l] + x3r[l], x1i[l] + x3i[l]);
      let (b1r, b1i) = (x1r[l] - x3r[l], x1i[l] - x3i[l]);
      gr[0][l] = a0r + b0r;
      gi[0][l] = a0i + b0i;
      gr[1][l] = a1r + b1i;
      gi[1][l] = a1i - b1r;
      gr[2][l] = a0r - b0r;
      gi[2][l] = a0i - b0i;
      gr[3][l] = a1r - b1i;
      gi[3][l] = a1i + b1r;
    }
  }
}

#[inline(never)]
fn stage_r4_f64(re: &mut [[f64; 8]], im: &mut [[f64; 8]], stride: usize, tw: &[[f64; 6]]) {
  let tw = &tw[..stride];
  for (gr, gi) in re.chunks_exact_mut(4 * stride).zip(im.chunks_exact_mut(4 * stride)) {
    let (r0, rest) = gr.split_at_mut(stride);
    let (r1, rest) = rest.split_at_mut(stride);
    let (r2, r3) = rest.split_at_mut(stride);
    let (i0, rest) = gi.split_at_mut(stride);
    let (i1, rest) = rest.split_at_mut(stride);
    let (i2, i3) = rest.split_at_mut(stride);
    for (u, &[c1, s1, c2, s2, c3, s3]) in tw.iter().enumerate() {
      let (x0r, x1r, x2r, x3r) = (r0[u], r1[u], r2[u], r3[u]);
      let (x0i, x1i, x2i, x3i) = (i0[u], i1[u], i2[u], i3[u]);
      let (mut y0r, mut y1r, mut y2r, mut y3r) = ([0f64; 8], [0f64; 8], [0f64; 8], [0f64; 8]);
      let (mut y0i, mut y1i, mut y2i, mut y3i) = ([0f64; 8], [0f64; 8], [0f64; 8], [0f64; 8]);
      for l in 0..8 {
        let (z0r, z0i) = (x0r[l], x0i[l]);
        let (z1r, z1i) = (x1i[l].mul_add(s1, x1r[l] * c1), (-x1r[l]).mul_add(s1, x1i[l] * c1));
        let (z2r, z2i) = (x2i[l].mul_add(s2, x2r[l] * c2), (-x2r[l]).mul_add(s2, x2i[l] * c2));
        let (z3r, z3i) = (x3i[l].mul_add(s3, x3r[l] * c3), (-x3r[l]).mul_add(s3, x3i[l] * c3));
        let (a0r, a0i) = (z0r + z2r, z0i + z2i);
        let (a1r, a1i) = (z0r - z2r, z0i - z2i);
        let (b0r, b0i) = (z1r + z3r, z1i + z3i);
        let (b1r, b1i) = (z1r - z3r, z1i - z3i);
        y0r[l] = a0r + b0r;
        y0i[l] = a0i + b0i;
        y1r[l] = a1r + b1i;
        y1i[l] = a1i - b1r;
        y2r[l] = a0r - b0r;
        y2i[l] = a0i - b0i;
        y3r[l] = a1r - b1i;
        y3i[l] = a1i + b1r;
      }
      r0[u] = y0r;
      r1[u] = y1r;
      r2[u] = y2r;
      r3[u] = y3r;
      i0[u] = y0i;
      i1[u] = y1i;
      i2[u] = y2i;
      i3[u] = y3i;
    }
  }
}

#[cfg(test)]
mod tests {
  use super::{Rfft8, Rfft8F64};

  fn check_rfft8(n: usize) {
    let frames: [Vec<f32>; 8] = std::array::from_fn(|l| {
      (0..n).map(|i| (((i + l * 37) * 7919 % 1000) as f32 / 500.0 - 1.0) * 0.3).collect()
    });
    let window: Vec<f32> = (0..n)
      .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos()) as f32)
      .collect();
    let m = n / 2;
    let mut re = vec![[0f32; 8]; m];
    let mut im = vec![[0f32; 8]; m];
    let mut power = vec![[0f32; 8]; m + 1];
    let refs: [&[f32]; 8] = std::array::from_fn(|l| &frames[l][..]);
    Rfft8::new(n).power_windowed(refs, &window, &mut re, &mut im, &mut power);
    for l in 0..8 {
      for (k, p) in power.iter().enumerate() {
        let (mut want_re, mut want_im) = (0f64, 0f64);
        for i in 0..n {
          let v = frames[l][i] as f64 * window[i] as f64;
          let phase = std::f64::consts::TAU * (k * i % n) as f64 / n as f64;
          want_re += v * phase.cos();
          want_im -= v * phase.sin();
        }
        let want = want_re * want_re + want_im * want_im;
        let got = p[l] as f64;
        assert!((got - want).abs() < 1e-3, "Rfft8 n={n} frame={l} bin={k}: got {got}, want {want}");
      }
    }
  }

  #[test]
  fn matches_a_dft() {
    for n in [320, 400, 512] {
      check_rfft8(n);
    }
    let n = 512;
    let frames: [Vec<f64>; 8] = std::array::from_fn(|l| {
      (0..n).map(|i| (((i + l * 37) * 7919 % 1000) as f64 / 500.0 - 1.0) * 0.3).collect()
    });
    let window: Vec<f64> =
      (0..n).map(|i| 0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / n as f64).cos()).collect();
    let (mut re, mut im, mut power) =
      (vec![[0f64; 8]; n / 2], vec![[0f64; 8]; n / 2], vec![[0f32; 8]; n / 2 + 1]);
    let refs: [&[f64]; 8] = std::array::from_fn(|l| &frames[l][..]);
    Rfft8F64::new(n).power_windowed_f32(refs, &window, &mut re, &mut im, &mut power);
    for l in 0..8 {
      for (k, p) in power.iter().enumerate() {
        let (mut want_re, mut want_im) = (0f64, 0f64);
        for i in 0..n {
          let v = frames[l][i] * window[i];
          let phase = std::f64::consts::TAU * (k * i % n) as f64 / n as f64;
          want_re += v * phase.cos();
          want_im -= v * phase.sin();
        }
        let want = (want_re * want_re + want_im * want_im) as f32;
        assert!((p[l] - want).abs() < 1e-5, "Rfft8F64 frame={l} bin={k}");
      }
    }
  }

  #[test]
  fn fast_ln_accuracy() {
    let mut max_ulp = 0u32;
    let mut max_abs = 0f32;
    let mut x = 1e-10f32;
    while x <= 1e10 {
      let got = crate::cpu::ops::ln(x);
      let want = x.ln();
      max_abs = max_abs.max((got - want).abs());
      if got.is_sign_positive() == want.is_sign_positive() {
        max_ulp = max_ulp.max(got.to_bits().abs_diff(want.to_bits()));
      }
      x *= 1.0001;
    }
    assert!(max_ulp <= 3 && max_abs < 2e-6, "ln max_ulp={max_ulp}, max_abs={max_abs:e}");
  }
}
