// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Log-mel features as transcribe.cpp computes them for Parakeet (`src/transcribe-mel.cpp`):
//! pre-emphasis in f64 before 256 zeros of padding on each side, symmetric Hann(400) centred in an
//! f64 FFT of 512, power cast to f32, Slaney filterbank, `ln(mel + 2⁻²⁴)`, then each mel bin
//! normalized over all frames but the last, which is zero. The diarizer takes the log-mel without
//! the normalization. The GGUF's dither is never applied.

use std::f64::consts::PI;
use std::ops::Range;

use crate::cpu::ops;
use crate::fft::Rfft8F64;

pub const MELS: usize = 128;
pub const N_FFT: usize = 512;
pub const WINDOW: usize = 400;
pub const HOP: usize = 160;
pub const PRE_EMPHASIS: f32 = 0.97;
pub const F_MAX: f32 = 8000.0;
const BINS: usize = N_FFT / 2 + 1;
const LOG_EPS: f32 = 1.0 / (1 << 24) as f32;
/// Frames per L1-resident pre-emphasis tile (~44 KiB in `f64`, plus ~40 KiB FFT scratch).
const TILE_FRAMES: usize = 32;
/// Minimum frames before splitting across 4 threads beats thread-spawn overhead.
const PAR_MIN_FRAMES: usize = 256;

pub struct Frontend {
  window: Vec<f64>,
  /// Concatenated nonzero weights of every mel filter, in filter order.
  weights: Vec<f32>,
  /// Per filter, `(first_bin, len)` of its nonzero weights.
  spans: Vec<(u16, u16)>,
  fft: Rfft8F64,
}

impl Frontend {
  pub fn new() -> Self {
    let pad = (N_FFT - WINDOW) / 2;
    let window = (0..N_FFT)
      .map(|i| match i.wrapping_sub(pad) {
        k if k < WINDOW => 0.5 - 0.5 * (2.0 * PI * k as f64 / (WINDOW - 1) as f64).cos(),
        _ => 0.0,
      })
      .collect();
    let filters = slaney_filterbank();
    let mut weights = Vec::new();
    let mut spans = Vec::with_capacity(MELS);
    for f in filters.as_chunks::<BINS>().0 {
      if let (Some(lo), Some(hi)) =
        (f.iter().position(|&w| w != 0.0), f.iter().rposition(|&w| w != 0.0))
      {
        let len = hi + 1 - lo;
        weights.extend_from_slice(&f[lo..lo + len]);
        spans.push((lo as u16, len as u16));
      } else {
        spans.push((0, 0));
      }
    }
    Self { window, weights, spans, fft: Rfft8F64::new(N_FFT) }
  }

  /// Features of 16 kHz PCM, time-major `[pcm.len() / 160 + 1][MELS]`; empty below 3 frames, too
  /// few for the normalization.
  pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
    let frames = pcm.len() / HOP + 1;
    if frames < 3 {
      return Vec::new();
    }

    let mut mel = self.log_mel(pcm, 0..frames);
    // Unbiased statistics over all frames but the last, row-major so every pass vectorizes.
    let n = frames - 1;
    let (body, last) = mel.split_at_mut(n * MELS);
    let mut mean = [0.0f64; MELS];
    for row in body.chunks_exact(MELS) {
      for (m, &x) in mean.iter_mut().zip(row) {
        *m += x as f64;
      }
    }
    let inv_n = 1.0 / n as f64;
    for m in &mut mean {
      *m *= inv_n;
    }
    let mut var = [0.0f64; MELS];
    for row in body.chunks_exact(MELS) {
      for ((v, &m), &x) in var.iter_mut().zip(&mean).zip(row) {
        let d = x as f64 - m;
        *v += d * d;
      }
    }
    let inv_nm1 = 1.0 / (n - 1) as f64;
    let eps = 1e-5f32 as f64;
    let mut inv = [0.0f64; MELS];
    for (iv, &v) in inv.iter_mut().zip(&var) {
      *iv = 1.0 / ((v * inv_nm1).sqrt() + eps);
    }
    for row in body.chunks_exact_mut(MELS) {
      for ((x, &m), &iv) in row.iter_mut().zip(&mean).zip(&inv) {
        *x = ((*x as f64 - m) * iv) as f32;
      }
    }
    last.fill(0.0);
    mel
  }

  /// The log-mel of `frames`, time-major `[frames.len()][MELS]`: frames of the whole `pcm`, up to
  /// `pcm.len() / 160 + 1` of them, computed from only the samples they cover.
  pub fn log_mel(&self, pcm: &[f32], frames: Range<usize>) -> Vec<f32> {
    let n_frames = frames.len();
    if n_frames == 0 {
      return Vec::new();
    }
    let mut mel = vec![0f32; n_frames * MELS];
    if n_frames >= PAR_MIN_FRAMES {
      let n_threads = crate::cpu::p_cores().clamp(1, 4);
      let per_frames = n_frames.div_ceil(n_threads).next_multiple_of(8);
      let per = per_frames * MELS;
      let chunks: Vec<(usize, &mut [f32])> =
        mel.chunks_mut(per).enumerate().map(|(i, s)| (frames.start + i * per_frames, s)).collect();
      crate::cpu::par_each(
        chunks.into_iter(),
        || (),
        |_, (f0, slice)| {
          self.log_mel_range(pcm, f0, slice);
        },
      );
    } else {
      self.log_mel_range(pcm, frames.start, &mut mel);
    }
    mel
  }

  fn log_mel_range(&self, pcm: &[f32], frame0: usize, out: &mut [f32]) {
    let total_frames = out.len() / MELS;
    let tile_cap = (TILE_FRAMES.min(total_frames) - 1) * HOP + N_FFT;
    let mut padded = vec![0.0f64; tile_cap];
    let (mut re, mut im) = ([[0f64; 8]; N_FFT / 2], [[0f64; 8]; N_FFT / 2]);
    let mut power = [[0f32; 8]; BINS];
    let half = N_FFT / 2;
    let pe = PRE_EMPHASIS as f64;

    for (tile_idx, tile_out) in out.chunks_mut(TILE_FRAMES * MELS).enumerate() {
      let tile_start = frame0 + tile_idx * TILE_FRAMES;
      let tile_frames = tile_out.len() / MELS;
      let pad = tile_start * HOP;
      let len = (tile_frames - 1) * HOP + N_FFT;
      let buf = &mut padded[..len];

      let mut fill_lo = 0usize;
      let mut fill_hi = 0usize;
      if !pcm.is_empty() {
        let i_lo = pad.saturating_sub(half).max(1);
        let i_hi = (pad + len).saturating_sub(half).min(pcm.len());
        if i_lo < i_hi {
          fill_lo = i_lo + half - pad;
          fill_hi = i_hi + half - pad;
          let dst = &mut buf[fill_lo..fill_hi];
          for ((d, &cur), &prev) in
            dst.iter_mut().zip(&pcm[i_lo..i_hi]).zip(&pcm[i_lo - 1..i_hi - 1])
          {
            *d = cur as f64 - pe * prev as f64;
          }
        }
      }
      buf[..fill_lo].fill(0.0);
      buf[fill_hi..].fill(0.0);
      if !pcm.is_empty() && pad <= half && half < pad + len {
        buf[half - pad] = pcm[0] as f64;
      }

      for (batch, chunk) in tile_out.chunks_mut(8 * MELS).enumerate() {
        let base = batch * 8;
        let count = chunk.len() / MELS;
        let last = count - 1;
        let fs = [
          &buf[base * HOP..][..N_FFT],
          &buf[(base + 1.min(last)) * HOP..][..N_FFT],
          &buf[(base + 2.min(last)) * HOP..][..N_FFT],
          &buf[(base + 3.min(last)) * HOP..][..N_FFT],
          &buf[(base + 4.min(last)) * HOP..][..N_FFT],
          &buf[(base + 5.min(last)) * HOP..][..N_FFT],
          &buf[(base + 6.min(last)) * HOP..][..N_FFT],
          &buf[(base + 7.min(last)) * HOP..][..N_FFT],
        ];
        self.fft.power_windowed_f32(fs, &self.window, &mut re, &mut im, &mut power);
        ops::mel_filter8_f32(&self.weights, &self.spans, &power, chunk, count);
        ops::ln_add_inplace(chunk, LOG_EPS);
      }
    }
  }
}

/// librosa's `mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000, norm='slaney')` as
/// `build_mel_filterbank_slaney` in transcribe.cpp's `src/transcribe-mel.cpp` computes it.
fn slaney_filterbank() -> Vec<f32> {
  const FSP: f64 = 200.0 / 3.0;
  const MIN_LOG_HZ: f64 = 1000.0;
  let min_log_mel = MIN_LOG_HZ / FSP;
  let log_step = 6.4f64.ln() / 27.0;
  let hz_to_mel = |hz: f64| {
    if hz < MIN_LOG_HZ { hz / FSP } else { min_log_mel + (hz / MIN_LOG_HZ).ln() / log_step }
  };
  let mel_to_hz = |mel: f64| {
    if mel < min_log_mel { mel * FSP } else { MIN_LOG_HZ * (log_step * (mel - min_log_mel)).exp() }
  };
  let mel_max = hz_to_mel(F_MAX as f64);
  let hz: Vec<f64> =
    (0..MELS + 2).map(|m| mel_to_hz(mel_max * m as f64 / (MELS + 1) as f64)).collect();
  let mut out = vec![0f32; MELS * BINS];
  for (m, row) in out.as_chunks_mut::<BINS>().0.iter_mut().enumerate() {
    let enorm = 2.0 / (hz[m + 2] - hz[m]);
    for (k, w) in row.iter_mut().enumerate() {
      let f = 16000.0 * k as f64 / N_FFT as f64;
      let lower = (f - hz[m]) / (hz[m + 1] - hz[m]);
      let upper = (hz[m + 2] - f) / (hz[m + 2] - hz[m + 1]);
      *w = (lower.min(upper).max(0.0) * enorm) as f32;
    }
  }
  out
}

#[cfg(test)]
mod tests {
  #[test]
  fn short_input() {
    let frontend = super::Frontend::new();
    assert!(frontend.compute(&[0.1; 319]).is_empty());
    let mel = frontend.compute(&[0.1; 320]);
    assert_eq!(mel.len(), 3 * super::MELS);
    assert!(mel.iter().all(|x| x.is_finite()));
  }

  /// A range of frames is the same as those frames of the whole audio, at the edges too.
  #[test]
  fn frame_ranges() {
    let frontend = super::Frontend::new();
    let pcm: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
    let all = frontend.log_mel(&pcm, 0..101);
    for r in [0..1, 0..7, 3..40, 95..101, 100..101] {
      let m = super::MELS;
      assert_eq!(frontend.log_mel(&pcm, r.clone()), all[r.start * m..r.end * m], "{r:?}");
    }
  }
}
