// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! The encoders' row-wise kernels on the CPU, as `src/gpu/ops.wgsl` and `attention.wgsl` compute
//! them, over time-major `[t][ch]` f32 activations, in plain arithmetic the compiler vectorizes.
//! Rows go to threads in chunks.

use half::f16;

use super::gemm::{NR, panels, product};
use super::{dot, par_each};
use crate::gpu::pos_rows;

/// `ln(x)` for positive normal `x` (`>= 1e-10`) as Cephes' `logf` computes it (error <= 2 ULP),
/// without branches or calls so loops of it autovectorize to NEON / AVX2. Uses Estrin's scheme
/// to halve the FMA dependency chain depth.
#[inline(always)]
pub fn ln(x: f32) -> f32 {
  const SQRT_HALF: u32 = 0x3f35_04f3;
  let bits = x.to_bits().wrapping_sub(SQRT_HALF);
  let e = (bits as i32 >> 23) as f32;
  let m = f32::from_bits((bits & 0x007f_ffff) + SQRT_HALF);
  let f = m - 1.0;
  let f2 = f * f;
  let f4 = f2 * f2;
  let p01 = (-2.499_999_4e-1f32).mul_add(f, 3.333_333e-1);
  let p23 = (-1.666_805_7e-1f32).mul_add(f, 2.000_071_5e-1);
  let p45 = (-1.242_014_1e-1f32).mul_add(f, 1.424_932_3e-1);
  let p67 = (-1.151_461e-1f32).mul_add(f, 1.167_699_9e-1);
  let p03 = p23.mul_add(f2, p01);
  let p47 = p67.mul_add(f2, p45);
  let p48 = 7.037_683_6e-2f32.mul_add(f4, p47);
  let p = p48.mul_add(f4, p03);
  let poly = (p * f).mul_add(f2, (-0.5f32).mul_add(f2, f));
  e.mul_add(0.693_359_4, e.mul_add(-2.121_944_4e-4, poly))
}

/// Applies `ln(x.clamp(lo, hi))` in place in 16-wide chunks so NEON constants stay in registers.
#[inline(never)]
pub fn ln_clamp_inplace(xs: &mut [f32], lo: f32, hi: f32) {
  let (chunks, tail) = xs.as_chunks_mut::<16>();
  for c in chunks {
    for v in c.iter_mut() {
      let x = if *v < lo {
        lo
      } else if *v > hi {
        hi
      } else {
        *v
      };
      *v = ln(x);
    }
  }
  for v in tail {
    let x = if *v < lo {
      lo
    } else if *v > hi {
      hi
    } else {
      *v
    };
    *v = ln(x);
  }
}

/// Applies `ln(x + eps)` in place in 16-wide chunks so NEON constants stay in registers.
#[inline(never)]
pub fn ln_add_inplace(xs: &mut [f32], eps: f32) {
  let (chunks, tail) = xs.as_chunks_mut::<16>();
  for c in chunks {
    for v in c.iter_mut() {
      *v = ln(*v + eps);
    }
  }
  for v in tail {
    *v = ln(*v + eps);
  }
}

/// Applies `log10(max(x, floor))` in place in 16-wide chunks so NEON constants stay in registers.
#[inline(never)]
pub fn log10_floor_inplace(xs: &mut [f32], floor: f32) {
  let (chunks, tail) = xs.as_chunks_mut::<16>();
  for c in chunks {
    for v in c.iter_mut() {
      let x = if *v < floor { floor } else { *v };
      *v = ln(x) * std::f32::consts::LOG10_E;
    }
  }
  for v in tail {
    let x = if *v < floor { floor } else { *v };
    *v = ln(x) * std::f32::consts::LOG10_E;
  }
}

/// Applies compact sparse `f32` mel filterbank (`weights`, `spans`) to eight power spectra
/// `power[bin][frame]` into `chunk` (`count` rows of `spans.len()` floats, `1..=8`), keeping
/// exact bin-order FMA per filter and 4×8 block transposes for contiguous stores.
pub fn mel_filter8_f32(
  weights: &[f32],
  spans: &[(u16, u16)],
  power: &[[f32; 8]],
  chunk: &mut [f32],
  count: usize,
) {
  let mels = spans.len();
  debug_assert!(chunk.len() == count * mels && (1..=8).contains(&count));

  let mut off = 0usize;
  if count == 8 {
    let (r0, rest) = chunk.split_at_mut(mels);
    let (r1, rest) = rest.split_at_mut(mels);
    let (r2, rest) = rest.split_at_mut(mels);
    let (r3, rest) = rest.split_at_mut(mels);
    let (r4, rest) = rest.split_at_mut(mels);
    let (r5, rest) = rest.split_at_mut(mels);
    let (r6, r7) = rest.split_at_mut(mels);
    let (spans4, spans_tail) = spans.as_chunks::<4>();
    let ((d0, t0), (d1, t1)) = (r0.as_chunks_mut::<4>(), r1.as_chunks_mut::<4>());
    let ((d2, t2), (d3, t3)) = (r2.as_chunks_mut::<4>(), r3.as_chunks_mut::<4>());
    let ((d4, t4), (d5, t5)) = (r4.as_chunks_mut::<4>(), r5.as_chunks_mut::<4>());
    let ((d6, t6), (d7, t7)) = (r6.as_chunks_mut::<4>(), r7.as_chunks_mut::<4>());
    for (idx, sp) in spans4.iter().enumerate() {
      let mut blk = [[0f32; 8]; 4];
      for (slot, &(lo, len)) in blk.iter_mut().zip(sp) {
        let (lo, len) = (lo as usize, len as usize);
        let ws = &weights[off..off + len];
        off += len;
        let mut e = [0f32; 8];
        for (&w, p) in ws.iter().zip(&power[lo..lo + len]) {
          for (a, &p) in e.iter_mut().zip(p) {
            *a = w.mul_add(p, *a);
          }
        }
        *slot = e;
      }
      d0[idx] = [blk[0][0], blk[1][0], blk[2][0], blk[3][0]];
      d1[idx] = [blk[0][1], blk[1][1], blk[2][1], blk[3][1]];
      d2[idx] = [blk[0][2], blk[1][2], blk[2][2], blk[3][2]];
      d3[idx] = [blk[0][3], blk[1][3], blk[2][3], blk[3][3]];
      d4[idx] = [blk[0][4], blk[1][4], blk[2][4], blk[3][4]];
      d5[idx] = [blk[0][5], blk[1][5], blk[2][5], blk[3][5]];
      d6[idx] = [blk[0][6], blk[1][6], blk[2][6], blk[3][6]];
      d7[idx] = [blk[0][7], blk[1][7], blk[2][7], blk[3][7]];
    }
    for (m, &(lo, len)) in spans_tail.iter().enumerate() {
      let (lo, len) = (lo as usize, len as usize);
      let ws = &weights[off..off + len];
      off += len;
      let mut e = [0f32; 8];
      for (&w, p) in ws.iter().zip(&power[lo..lo + len]) {
        for (a, &p) in e.iter_mut().zip(p) {
          *a = w.mul_add(p, *a);
        }
      }
      t0[m] = e[0];
      t1[m] = e[1];
      t2[m] = e[2];
      t3[m] = e[3];
      t4[m] = e[4];
      t5[m] = e[5];
      t6[m] = e[6];
      t7[m] = e[7];
    }
  } else {
    for (m, &(lo, len)) in spans.iter().enumerate() {
      let (lo, len) = (lo as usize, len as usize);
      let ws = &weights[off..off + len];
      off += len;
      let mut e = [0f32; 8];
      for (&w, p) in ws.iter().zip(&power[lo..lo + len]) {
        for (a, &p) in e.iter_mut().zip(p) {
          *a = w.mul_add(p, *a);
        }
      }
      for (l, row) in chunk.chunks_exact_mut(mels).enumerate() {
        row[m] = e[l];
      }
    }
  }
}

/// Applies compact sparse `f64` mel filterbank (`weights`, `spans`) to eight `f64` power spectra
/// `power[bin][frame]`, storing `f32` outputs into `chunk` (`count` rows of `spans.len()` floats).
pub fn mel_filter8_f64(
  weights: &[f64],
  spans: &[(u16, u16)],
  power: &[[f64; 8]],
  chunk: &mut [f32],
  count: usize,
) {
  let mels = spans.len();
  debug_assert!(chunk.len() == count * mels && (1..=8).contains(&count));

  let mut off = 0usize;
  if count == 8 {
    let (r0, rest) = chunk.split_at_mut(mels);
    let (r1, rest) = rest.split_at_mut(mels);
    let (r2, rest) = rest.split_at_mut(mels);
    let (r3, rest) = rest.split_at_mut(mels);
    let (r4, rest) = rest.split_at_mut(mels);
    let (r5, rest) = rest.split_at_mut(mels);
    let (r6, r7) = rest.split_at_mut(mels);
    let (spans4, spans_tail) = spans.as_chunks::<4>();
    let ((d0, t0), (d1, t1)) = (r0.as_chunks_mut::<4>(), r1.as_chunks_mut::<4>());
    let ((d2, t2), (d3, t3)) = (r2.as_chunks_mut::<4>(), r3.as_chunks_mut::<4>());
    let ((d4, t4), (d5, t5)) = (r4.as_chunks_mut::<4>(), r5.as_chunks_mut::<4>());
    let ((d6, t6), (d7, t7)) = (r6.as_chunks_mut::<4>(), r7.as_chunks_mut::<4>());
    for (idx, sp) in spans4.iter().enumerate() {
      let mut blk = [[0f32; 8]; 4];
      for (slot, &(lo, len)) in blk.iter_mut().zip(sp) {
        let (lo, len) = (lo as usize, len as usize);
        let ws = &weights[off..off + len];
        off += len;
        let mut s = [0.0f64; 8];
        for (&w, p) in ws.iter().zip(&power[lo..lo + len]) {
          for (a, &p) in s.iter_mut().zip(p) {
            *a = w.mul_add(p, *a);
          }
        }
        for (dst, v) in slot.iter_mut().zip(s) {
          *dst = v as f32;
        }
      }
      d0[idx] = [blk[0][0], blk[1][0], blk[2][0], blk[3][0]];
      d1[idx] = [blk[0][1], blk[1][1], blk[2][1], blk[3][1]];
      d2[idx] = [blk[0][2], blk[1][2], blk[2][2], blk[3][2]];
      d3[idx] = [blk[0][3], blk[1][3], blk[2][3], blk[3][3]];
      d4[idx] = [blk[0][4], blk[1][4], blk[2][4], blk[3][4]];
      d5[idx] = [blk[0][5], blk[1][5], blk[2][5], blk[3][5]];
      d6[idx] = [blk[0][6], blk[1][6], blk[2][6], blk[3][6]];
      d7[idx] = [blk[0][7], blk[1][7], blk[2][7], blk[3][7]];
    }
    for (m, &(lo, len)) in spans_tail.iter().enumerate() {
      let (lo, len) = (lo as usize, len as usize);
      let ws = &weights[off..off + len];
      off += len;
      let mut s = [0.0f64; 8];
      for (&w, p) in ws.iter().zip(&power[lo..lo + len]) {
        for (a, &p) in s.iter_mut().zip(p) {
          *a = w.mul_add(p, *a);
        }
      }
      t0[m] = s[0] as f32;
      t1[m] = s[1] as f32;
      t2[m] = s[2] as f32;
      t3[m] = s[3] as f32;
      t4[m] = s[4] as f32;
      t5[m] = s[5] as f32;
      t6[m] = s[6] as f32;
      t7[m] = s[7] as f32;
    }
  } else {
    for (m, &(lo, len)) in spans.iter().enumerate() {
      let (lo, len) = (lo as usize, len as usize);
      let ws = &weights[off..off + len];
      off += len;
      let mut s = [0.0f64; 8];
      for (&w, p) in ws.iter().zip(&power[lo..lo + len]) {
        for (a, &p) in s.iter_mut().zip(p) {
          *a = w.mul_add(p, *a);
        }
      }
      for (l, row) in chunk.chunks_exact_mut(mels).enumerate() {
        row[m] = s[l] as f32;
      }
    }
  }
}

/// e^x as Cephes' `expf` computes it (relative error ~1e-7) without branches or calls, using
/// Estrin's scheme with `mul_add` so loops autovectorize to low-latency NEON / AVX2 FMAs.
#[inline(always)]
pub fn exp(x: f32) -> f32 {
  const MAGIC: f32 = 12_582_912.0;
  let x = x.clamp(-87.0, 88.0);
  let t = x.mul_add(std::f32::consts::LOG2_E, MAGIC);
  let n = t - MAGIC;
  let r = n.mul_add(2.121_944_4e-4, n.mul_add(-0.693_359_4, x));
  let r2 = r * r;
  let p01 = 0.166_666_65f32.mul_add(r, 0.5);
  let p23 = 8.333_452e-3f32.mul_add(r, 4.166_579_6e-2);
  let p45 = 1.987_569_1e-4f32.mul_add(r, 1.398_199_9e-3);
  let p = p45.mul_add(r2, p23).mul_add(r2, p01).mul_add(r2, r) + 1.0;
  let scale = f32::from_bits(((t.to_bits() as i32 - 0x4B40_0000 + 127) << 23) as u32);
  p * scale
}

#[inline(always)]
pub fn silu(x: f32) -> f32 {
  x / (1.0 + exp(-x))
}

/// GELU with erf as the GPU's `gelu` computes it: Abramowitz–Stegun 7.1.26, as ggml-metal.
#[inline(always)]
pub fn gelu(x: f32) -> f32 {
  let z = (std::f32::consts::FRAC_1_SQRT_2 * x).abs();
  let t = 1.0 / 0.327_591_1f32.mul_add(z, 1.0);
  let t2 = t * t;
  let p01 = (-0.284_496_74f32).mul_add(t, 0.254_829_6);
  let p23 = (-1.453_152f32).mul_add(t, 1.421_413_8);
  let p = 1.061_405_4f32.mul_add(t2, p23).mul_add(t2, p01);
  let s = 1.0f32.copysign(x);
  0.5 * x * (1.0 + s * (1.0 - p * t * exp(-z * z)))
}

/// Rows per job for `rows` rows: about four jobs per thread.
fn chunk(rows: usize) -> usize {
  rows.div_ceil(4 * super::threads()).max(1)
}

/// Pairwise reduction of 16 lanes in 4-float vectors so LLVM emits full 128-bit
/// `fadd.4s`/`fmla.4s`.
#[inline]
fn reduce16(acc: [f32; 16]) -> f32 {
  let mut s = [0.0f32; 4];
  for i in 0..4 {
    s[i] = (acc[i] + acc[i + 4]) + (acc[i + 8] + acc[i + 12]);
  }
  s[0] + s[1] + s[2] + s[3]
}

/// Mean and 1/√(variance + 1e-5) of a row, vectorized in 16-float chunks.
fn moments(x: &[f32]) -> (f32, f32) {
  let n = x.len() as f32;
  let mean = sum_slice(x) / n;
  let (chunks, tail) = x.as_chunks::<16>();
  let mut acc = [0.0f32; 16];
  for c in chunks {
    for (s, &v) in acc.iter_mut().zip(c) {
      let d = v - mean;
      *s = d.mul_add(d, *s);
    }
  }
  let mut var = reduce16(acc);
  for &v in tail {
    let d = v - mean;
    var = d.mul_add(d, var);
  }
  (mean, 1.0 / (var / n + 1e-5).sqrt())
}

/// A sum of a slice in 16 lanes (four NEON vectors), which autovectorizes cleanly.
#[inline]
fn sum_slice(x: &[f32]) -> f32 {
  let (chunks, tail) = x.as_chunks::<16>();
  let mut acc = [0.0f32; 16];
  for c in chunks {
    for (s, &v) in acc.iter_mut().zip(c) {
      *s += v;
    }
  }
  reduce16(acc) + tail.iter().sum::<f32>()
}

/// LayerNorm (eps 1e-5) of the rows of `x` into `y`, both `[rows][w.len()]`; with `rope` (a table
/// `[t][cos | sin]` of head_dim / 2 frequencies each, `yr`, head_dim) also the same with rotary
/// embedding (NEOX split-half rotation inside each head) into `yr`.
pub fn layer_norm(
  x: &[f32],
  (w, b): (&[f32], &[f32]),
  y: &mut [f32],
  rope: Option<(&[f32], &mut [f32], usize)>,
) {
  let cols = w.len();
  let rows = chunk(x.len() / cols);
  let norm = |x: &[f32], y: &mut [f32]| {
    for (x, y) in x.chunks_exact(cols).zip(y.chunks_exact_mut(cols)) {
      let (mean, rstd) = moments(x);
      for (((y, x), w), b) in y.iter_mut().zip(x).zip(w).zip(b) {
        *y = (x - mean) * rstd * w + b;
      }
    }
  };
  let Some((table, yr, hd)) = rope else {
    let jobs = x.chunks(rows * cols).zip(y.chunks_mut(rows * cols));
    return par_each(jobs, || (), |_, (x, y)| norm(x, y));
  };
  let half = hd / 2;
  let jobs = x.chunks(rows * cols).zip(y.chunks_mut(rows * cols)).zip(yr.chunks_mut(rows * cols));
  let jobs = jobs.zip(table.chunks(rows * hd));
  par_each(
    jobs,
    || (),
    |_, (((x, y), yr), table)| {
      norm(x, y);
      for ((y, yr), t) in y.chunks_exact(cols).zip(yr.chunks_exact_mut(cols)).zip(table.chunks(hd))
      {
        let (cos, sin) = t.split_at(half);
        for (y, yr) in y.chunks_exact(hd).zip(yr.chunks_exact_mut(hd)) {
          let (y0, y1) = y.split_at(half);
          let (r0, r1) = yr.split_at_mut(half);
          for i in 0..half {
            r0[i] = y0[i] * cos[i] - y1[i] * sin[i];
            r1[i] = y1[i] * cos[i] + y0[i] * sin[i];
          }
        }
      }
    },
  );
}

/// Columns for a conv1d with `[kernel, stride, pad]` over `x` `[t][ch]`: rows of `ch · kernel`,
/// `col[i][c · kernel + k] = x[stride · i + k − pad][c]` (a ggml conv weight `[kernel, ch, out]`).
pub fn im2col(x: &[f32], t: usize, ch: usize, [kernel, stride, pad]: [usize; 3]) -> Vec<f32> {
  let rows = (t + 2 * pad - kernel) / stride + 1;
  let mut col = vec![0.0; rows * ch * kernel];
  if ch == 1 && pad == 0 {
    for (i, dst) in col.chunks_exact_mut(kernel).enumerate() {
      dst.copy_from_slice(&x[stride * i..][..kernel]);
    }
    return col;
  }
  if kernel == 3 {
    for (i, col) in col.chunks_exact_mut(ch * 3).enumerate() {
      let s0 = stride * i;
      if s0 >= pad && s0 - pad + 3 <= t {
        let base = (s0 - pad) * ch;
        let (r0, r1, r2) =
          (&x[base..base + ch], &x[base + ch..base + 2 * ch], &x[base + 2 * ch..base + 3 * ch]);
        for (dst, ((&a, &b), &c)) in
          col.as_chunks_mut::<3>().0.iter_mut().zip(r0.iter().zip(r1).zip(r2))
        {
          *dst = [a, b, c];
        }
      } else {
        for k in 0..3 {
          let s = s0 + k;
          if s < pad || s - pad >= t {
            continue;
          }
          for (dst, &v) in col.as_chunks_mut::<3>().0.iter_mut().zip(&x[(s - pad) * ch..][..ch]) {
            dst[k] = v;
          }
        }
      }
    }
    return col;
  }
  for (i, col) in col.chunks_exact_mut(ch * kernel).enumerate() {
    for k in 0..kernel {
      let s = stride * i + k;
      if s < pad || s - pad >= t {
        continue;
      }

      for (c, x) in x[(s - pad) * ch..][..ch].iter().enumerate() {
        col[c * kernel + k] = *x;
      }
    }
  }

  col
}

/// The Conformer conv module between the pointwise convs: GLU (first half · sigmoid(second half))
/// of `h` `[t][2·ch]` in place into its first halves, the depthwise conv (odd kernel, centred,
/// `dw` taps `[kernel][ch]` and bias), LayerNorm over channels (`layer_norm`) or a per-channel
/// affine `(g, b)`, SiLU, into `y` `[t][ch]`.
pub fn conv_glu(
  h: &mut [f32],
  (taps, bias): (&[f32], &[f32]),
  (g, beta): (&[f32], &[f32]),
  layer_norm: bool,
  y: &mut [f32],
) {
  let ch = bias.len();
  let (t, kernel) = (y.len() / ch, taps.len() / ch);
  let rows = chunk(t);
  par_each(
    h.chunks_mut(rows * 2 * ch),
    || (),
    |_, h| {
      for h in h.chunks_exact_mut(2 * ch) {
        let (x, gate) = h.split_at_mut(ch);
        for (x, gate) in x.iter_mut().zip(&*gate) {
          *x /= 1.0 + exp(-gate);
        }
      }
    },
  );
  let h = &*h;
  par_each(
    y.chunks_mut(rows * ch).enumerate(),
    || vec![0.0; ch],
    |acc, (j, y)| {
      for (i, y) in y.chunks_exact_mut(ch).enumerate() {
        let i = j * rows + i;
        acc.copy_from_slice(bias);
        for (k, taps) in taps.chunks_exact(ch).enumerate() {
          // Source frame plus the padding.
          let Some(src) = (i + k).checked_sub(kernel / 2).filter(|&s| s < t) else { continue };
          for ((a, w), x) in acc.iter_mut().zip(taps).zip(&h[src * 2 * ch..][..ch]) {
            *a = w.mul_add(*x, *a);
          }
        }

        let (mean, rstd) = if layer_norm { moments(acc) } else { (0.0, 1.0) };
        for (((y, a), g), b) in y.iter_mut().zip(&*acc).zip(g).zip(beta) {
          *y = silu((a - mean) * rstd * g + b);
        }
      }
    },
  );
}

/// Query rows per step of `attention`: scores for 48 queries over 1500 keys take 280 KiB.
const QUERIES: usize = 48;

/// Multi-head attention with any rotary positions already applied: row i's q at `qk[i · ld..]`,
/// its k at `qk[i · ld + ch..]` and its v at `v[i · ldv..]`, into `o` `[t][ch]`, scores scaled by
/// 1/√head_dim. One head per job.
pub fn attention(
  qk: &[f32],
  ld: usize,
  v: &[f32],
  ldv: usize,
  o: &mut [f32],
  t: usize,
  heads: usize,
) {
  let ch = o.len() / t;
  let (hd, tp) = (ch / heads, t.next_multiple_of(NR));
  // A head's output rows are as wide as its value panels.
  let hp = hd.next_multiple_of(NR);
  let scale = 1.0 / (hd as f32).sqrt();
  let mut outs = vec![Vec::new(); heads];
  par_each(
    outs.iter_mut().enumerate(),
    || vec![0.0; QUERIES * tp],
    |s, (h, out)| {
      let keys = panels(&qk[ch + h * hd..], ld, t, hd, hd, false);
      let values = panels(&v[h * hd..], ldv, hd, tp, t, true);
      *out = vec![0.0; t * hp];
      for i in (0..t).step_by(QUERIES) {
        let m = QUERIES.min(t - i);
        product(&qk[i * ld + h * hd..], ld, m, &keys, hd, s, tp);
        softmax(s, m, t, scale);
        product(s, tp, m, &values, tp, &mut out[i * hp..], hp);
      }
    },
  );
  gather(&outs, o, hd, hp);
}

/// Multi-head attention with Transformer-XL relative positions over `qkv` `[t][q | k | v]`, with
/// `pos` the projected positions `[pos_rows(t)][ch]` (row r is position t + 15 − r) and `uv` the
/// biases u and v `[heads][head_dim]`, into `o` `[t][ch]`: score (i, j) is
/// `((q + u)_i·k_j + (q + v)_i·pos[j − i + t + 15]) / √head_dim`. One head per job; heads of a
/// multiple of 16 dims.
pub fn relative_attention(
  qkv: &[f32],
  pos: &[f32],
  uv: (&[f32], &[f32]),
  o: &mut [f32],
  t: usize,
  heads: usize,
) {
  let ch = o.len() / t;
  let (hd, tp) = (ch / heads, t.next_multiple_of(NR));
  // A step's queries i..i + m read the position rows t + 16 − i − m..2t + 15 − i.
  let pw = (t + QUERIES + 2 * NR).next_multiple_of(NR);
  let scale = 1.0 / (hd as f32).sqrt();
  let mut outs = vec![Vec::new(); heads];
  par_each(
    outs.iter_mut().enumerate(),
    || (vec![0.0; QUERIES * tp], vec![0.0; QUERIES * pw]),
    |(s, sp), (h, out)| {
      let keys = panels(&qkv[ch + h * hd..], 3 * ch, t, hd, hd, false);
      let values = panels(&qkv[2 * ch + h * hd..], 3 * ch, hd, tp, t, true);
      let positions = panels(&pos[h * hd..], ch, pos_rows(t), hd, hd, false);
      let q = |b: &[f32]| -> Vec<f32> {
        let b = &b[h * hd..][..hd];
        let rows = qkv.chunks_exact(3 * ch).map(|r| &r[h * hd..][..hd]);
        rows.flat_map(|q| q.iter().zip(b).map(|(q, b)| q + b)).collect()
      };
      let (qu, qv) = (q(uv.0), q(uv.1));
      *out = vec![0.0; t * hd];
      for i in (0..t).step_by(QUERIES) {
        let m = QUERIES.min(t - i);
        product(&qu[i * hd..], hd, m, &keys, hd, s, tp);
        let (p0, p1) = ((t + 16 - i - m) / NR, (2 * t + 15 - i).div_ceil(NR));
        product(&qv[i * hd..], hd, m, &positions[p0 * NR * hd..p1 * NR * hd], hd, sp, pw);
        for (a, (s, sp)) in s.chunks_exact_mut(tp).zip(sp.chunks_exact(pw)).take(m).enumerate() {
          // Query i + a, key j: position row j − i − a + t + 15, from p0·NR on.
          let sp = &sp[t + 15 - i - a - p0 * NR..][..t];
          s.iter_mut().zip(sp).for_each(|(s, p)| *s += p);
        }

        softmax(s, m, t, scale);
        product(s, tp, m, &values, tp, &mut out[i * hd..], hd);
      }
    },
  );
  gather(&outs, o, hd, hd);
}

/// Softmax of `scores` × `scale` over the first `t` of each of `m` rows; the rest of a row is
/// zeroed.
#[inline(never)]
fn softmax(scores: &mut [f32], m: usize, t: usize, scale: f32) {
  let tp = t.next_multiple_of(NR);
  for row in scores.chunks_exact_mut(tp).take(m) {
    let (row, pad) = row.split_at_mut(t);
    let mut mmax = [f32::NEG_INFINITY; 16];
    let (chunks, tail) = row.as_chunks::<16>();
    for c in chunks {
      for (a, &v) in mmax.iter_mut().zip(c) {
        if v > *a {
          *a = v;
        }
      }
    }
    let mut max = mmax.into_iter().fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
    for &v in tail {
      if v > max {
        max = v;
      }
    }
    max *= scale;
    let (chunks, tail) = row.as_chunks_mut::<16>();
    let mut sum16 = [0.0f32; 16];
    for c in chunks {
      for (s, x) in sum16.iter_mut().zip(c.iter_mut()) {
        let e = exp((*x).mul_add(scale, -max));
        *x = e;
        *s += e;
      }
    }
    let mut total = reduce16(sum16);
    for x in tail {
      let e = exp((*x).mul_add(scale, -max));
      *x = e;
      total += e;
    }
    let inv = 1.0 / total;
    for x in row {
      *x *= inv;
    }
    pad.fill(0.0);
  }
}

/// The heads' outputs `[t][hp]` (the first head_dim of each row) into the rows
/// `[t][heads · head_dim]` of `o`.
fn gather(outs: &[Vec<f32>], o: &mut [f32], hd: usize, hp: usize) {
  let ch = outs.len() * hd;
  for (h, out) in outs.iter().enumerate() {
    for (o, out) in o.chunks_exact_mut(ch).zip(out.chunks_exact(hp)) {
      o[h * hd..][..hd].copy_from_slice(&out[..hd]);
    }
  }
}

/// One query per head, `q` `[heads][head_dim]` rounded to f16 as the GPU's `attend` rounds it,
/// over the first `n` positions of a cache `kv` (K, then V, each `[heads][cols][head_dim]`), into
/// `o` `[heads][head_dim]`; scores scaled by 1/√head_dim. Heads go to threads when the cache is
/// long.
pub fn attend(q: &[f32], kv: &[f32], cols: usize, n: usize, o: &mut [f32], heads: usize) {
  let hd = q.len() / heads;
  let scale = 1.0 / (hd as f32).sqrt();
  let (k, v) = kv.split_at(heads * cols * hd);
  // Jobs of at least 2^16 multiply-adds.
  let per = heads.div_ceil(((heads * n * hd) >> 16).clamp(1, heads));
  par_each(
    o.chunks_mut(per * hd).enumerate(),
    || (vec![0.0; hd], vec![0.0; n]),
    |(qh, s), (j, o)| {
      for (i, o) in o.chunks_exact_mut(hd).enumerate() {
        let h = j * per + i;
        for (qh, q) in qh.iter_mut().zip(&q[h * hd..][..hd]) {
          *qh = f16::from_f32(*q).to_f32();
        }

        let (k, v) = (&k[h * cols * hd..][..n * hd], &v[h * cols * hd..][..n * hd]);
        for (s, k) in s.iter_mut().zip(k.chunks_exact(hd)) {
          *s = dot(qh, k) * scale;
        }

        let max = s.iter().fold(f32::NEG_INFINITY, |m, &x| if x > m { x } else { m });
        let (chunks, tail) = s.as_chunks_mut::<16>();
        let mut sum16 = [0.0f32; 16];
        for c in chunks {
          for (a, x) in sum16.iter_mut().zip(c.iter_mut()) {
            let e = exp(*x - max);
            *x = e;
            *a += e;
          }
        }
        let mut total = reduce16(sum16);
        for x in tail {
          let e = exp(*x - max);
          *x = e;
          total += e;
        }
        let inv = 1.0 / total;
        o.fill(0.0);
        for (&s, v) in s.iter().zip(v.chunks_exact(hd)) {
          for (o, &v) in o.iter_mut().zip(v) {
            *o = s.mul_add(v, *o);
          }
        }

        for o in o.iter_mut() {
          *o *= inv;
        }
      }
    },
  );
}

/// Rotary embedding in place over rows `[q | k | v]` of `heads` heads of `hd`, row j at the
/// position of `table` row j (`rot` / 2 cosines, then as many sines): in q's and k's heads the
/// pairs (2i, 2i + 1) for 2i < `rot` rotate, or with `neox` (i, i + rot / 2), as the GPU's `rope`.
pub fn rope(qkv: &mut [f32], table: &[f32], [heads, hd, rot]: [usize; 3], neox: bool) {
  let half = rot / 2;
  for (row, t) in qkv.chunks_exact_mut(3 * heads * hd).zip(table.chunks_exact(rot)) {
    let (cos, sin) = t.split_at(half);
    for head in row[..2 * heads * hd].chunks_exact_mut(hd) {
      if neox {
        let (lo, hi) = head[..rot].split_at_mut(half);
        for (((a, b), &c), &s) in lo.iter_mut().zip(hi.iter_mut()).zip(cos).zip(sin) {
          let (x, y) = (*a, *b);
          *a = x * c - y * s;
          *b = x * s + y * c;
        }
      } else {
        for ((pair, &c), &s) in head[..rot].as_chunks_mut::<2>().0.iter_mut().zip(cos).zip(sin) {
          let (x, y) = (pair[0], pair[1]);
          pair[0] = x * c - y * s;
          pair[1] = x * s + y * c;
        }
      }
    }
  }
}

/// GroupNorm with one group in place over `x` `[t][ch]`: normalized over all its values (eps
/// `eps`), then times `w` and plus `b` per channel. The moments are summed in f64.
pub fn group_norm(x: &mut [f32], (w, b): (&[f32], &[f32]), eps: f32) {
  let n = x.len() as f64;
  let (chunks, tail) = x.as_chunks::<8>();
  let mut sum8 = [0.0f64; 8];
  for c in chunks {
    for (s, &v) in sum8.iter_mut().zip(c) {
      *s += v as f64;
    }
  }
  let mut sum =
    ((sum8[0] + sum8[4]) + (sum8[2] + sum8[6])) + ((sum8[1] + sum8[5]) + (sum8[3] + sum8[7]));
  for &v in tail {
    sum += v as f64;
  }
  let mean = sum / n;
  let mut var8 = [0.0f64; 8];
  for c in chunks {
    for (s, &v) in var8.iter_mut().zip(c) {
      let d = v as f64 - mean;
      *s = d.mul_add(d, *s);
    }
  }
  let mut var =
    ((var8[0] + var8[4]) + (var8[2] + var8[6])) + ((var8[1] + var8[5]) + (var8[3] + var8[7]));
  for &v in tail {
    let d = v as f64 - mean;
    var = d.mul_add(d, var);
  }
  let (mean, rstd) = (mean as f32, 1.0 / ((var / n) as f32 + eps).sqrt());
  for row in x.chunks_exact_mut(w.len()) {
    for ((x, w), b) in row.iter_mut().zip(w).zip(b) {
      *x = (*x - mean) * rstd * w + b;
    }
  }
}
