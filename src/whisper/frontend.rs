// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Log-mel features as transcribe.cpp computes them for Whisper (`src/transcribe-mel.cpp`): audio
//! up to 30 s zero-padded to 30 s, reflect padding of 200, periodic Hann window and Slaney
//! filterbank from the GGUF, an f32 mixed-radix FFT of 400, power, f64 mel sums, `log10(max(·,
//! 1e-10))`, the last frame dropped, then clamped to the global maximum − 8 and scaled as `(x + 4)
//! / 4`.

use crate::cpu::ops;
use crate::error::{Result, bail};
use crate::fft::Rfft8;
use crate::gguf::Gguf;

pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
const BINS: usize = N_FFT / 2 + 1;
/// A window: 30 s of samples, 3000 frames.
pub const SAMPLES: usize = 480_000;

pub struct Frontend {
  mels: usize,
  window: Vec<f32>,
  /// Concatenated nonzero weights of every mel filter, widened to f64 for the mel sum.
  weights: Vec<f64>,
  /// Per filter, `(first_bin, len)` of its nonzero weights.
  spans: Vec<(u16, u16)>,
  fft: Rfft8,
}

impl Frontend {
  pub fn new(g: &Gguf, mels: usize) -> Result<Self> {
    if mels == 0 {
      bail!("whisper frontend mel count must be nonzero");
    }
    let window = g.tensor("frontend.window", &[N_FFT])?.to_f32()?;
    let filters = g.tensor("frontend.mel_filterbank", &[BINS, mels])?.to_f32()?;
    let mut weights = Vec::new();
    let mut spans = Vec::with_capacity(mels);
    for f in filters.as_chunks::<BINS>().0 {
      if let (Some(lo), Some(hi)) =
        (f.iter().position(|&w| w != 0.0), f.iter().rposition(|&w| w != 0.0))
      {
        let len = hi + 1 - lo;
        weights.extend(f[lo..lo + len].iter().map(|&w| w as f64));
        spans.push((lo as u16, len as u16));
      } else {
        spans.push((0, 0));
      }
    }
    Ok(Self { mels, window, weights, spans, fft: Rfft8::new(N_FFT) })
  }

  /// Features `[frames][mels]`: 3000 frames for up to 30 s, else one per 160 samples.
  pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
    let len = pcm.len().max(SAMPLES);
    let pad = N_FFT / 2;
    let mut padded = vec![0f32; len + N_FFT];
    padded[pad..pad + pcm.len()].copy_from_slice(pcm);
    for i in 0..pad {
      padded[i] = padded[2 * pad - i];
      padded[pad + len + i] = padded[pad + len - 2 - i];
    }
    let frames = len / HOP;
    let mels = self.mels;
    let silent = 1e-10f64.log10() as f32;
    let mut mel = vec![silent; frames * mels];
    // Frames past the audio's padded end are zero (already filled with `silent`), so split only
    // the active prefix across four threads (3 spawned + current).
    let active = if pcm.is_empty() {
      0
    } else if pcm.len() + pad <= len {
      (pcm.len() + pad).div_ceil(HOP).min(frames)
    } else {
      frames
    };
    if active > 0 {
      let n_threads = crate::cpu::p_cores().clamp(1, 4);
      let per_frames = active.div_ceil(n_threads).next_multiple_of(8);
      let per = per_frames * mels;
      let active_mel = &mut mel[..active * mels];
      let padded = &padded;
      let chunks: Vec<(usize, &mut [f32])> =
        active_mel.chunks_mut(per).enumerate().map(|(i, out)| (i * per_frames, out)).collect();
      crate::cpu::par_each(
        chunks.into_iter(),
        || (),
        |_, (frame0, out)| {
          self.compute_range(padded, frame0, out, silent);
        },
      );
    }
    // Values are finite (clamped before log10), so a branchless lane max avoids NaN guards.
    let mut acc = [f32::NEG_INFINITY; 16];
    let (chunks, tail) = mel.as_chunks::<16>();
    for c in chunks {
      for (m, &v) in acc.iter_mut().zip(c) {
        if v > *m {
          *m = v;
        }
      }
    }
    let mut max = acc.into_iter().fold(f32::NEG_INFINITY, |a, b| if b > a { b } else { a });
    for &v in tail {
      if v > max {
        max = v;
      }
    }
    let floor = (max as f64 - 8.0) as f32;
    for v in &mut mel {
      let x = if *v > floor { *v } else { floor };
      *v = (x + 4.0) * 0.25;
    }
    mel
  }

  fn compute_range(&self, padded: &[f32], frame0: usize, out: &mut [f32], silent: f32) {
    let mels = self.mels;
    let (mut re, mut im) = ([[0f32; 8]; N_FFT / 2], [[0f32; 8]; N_FFT / 2]);
    let mut power = [[0f64; 8]; BINS];
    for (batch, chunk) in out.chunks_mut(8 * mels).enumerate() {
      let base = frame0 + batch * 8;
      let count = chunk.len() / mels;
      let last = count - 1;
      let fs = [
        &padded[base * HOP..][..N_FFT],
        &padded[(base + 1.min(last)) * HOP..][..N_FFT],
        &padded[(base + 2.min(last)) * HOP..][..N_FFT],
        &padded[(base + 3.min(last)) * HOP..][..N_FFT],
        &padded[(base + 4.min(last)) * HOP..][..N_FFT],
        &padded[(base + 5.min(last)) * HOP..][..N_FFT],
        &padded[(base + 6.min(last)) * HOP..][..N_FFT],
        &padded[(base + 7.min(last)) * HOP..][..N_FFT],
      ];
      // Skip the FFT when all eight frames in the batch are digital silence.
      let span = &padded[base * HOP..(base + last) * HOP + N_FFT];
      if is_all_zero(span) {
        chunk.fill(silent);
        continue;
      }
      self.fft.power_windowed(fs, &self.window, &mut re, &mut im, &mut power);
      ops::mel_filter8_f64(&self.weights, &self.spans, &power, chunk, count);
      ops::log10_floor_inplace(chunk, 1e-10);
    }
  }
}

/// Whether every sample in `x` is ±0.0, vectorized via bitwise OR of non-sign bits.
fn is_all_zero(x: &[f32]) -> bool {
  let (chunks, tail) = x.as_chunks::<16>();
  let mut acc = [0u32; 16];
  for c in chunks {
    for (a, &v) in acc.iter_mut().zip(c) {
      *a |= v.to_bits() & 0x7fff_ffff;
    }
  }
  let mut any = acc.into_iter().fold(0u32, |a, b| a | b);
  for &v in tail {
    any |= v.to_bits() & 0x7fff_ffff;
  }
  any == 0
}
