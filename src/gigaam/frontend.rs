// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! Log-mel features: periodic Hann window and HTK filterbank from the GGUF, FFT 320, hop 160,
//! full frames only (`center=False`), power spectrum, `ln(clamp(mel, 1e-9, 1e9))`.

use crate::cpu::ops;
use crate::error::{Result, bail};
use crate::fft::Rfft8;
use crate::gguf::Gguf;

pub const N_FFT: usize = 320;
pub const HOP: usize = 160;
const BINS: usize = N_FFT / 2 + 1;
/// Minimum frames before splitting across 4 threads beats thread-spawn overhead.
const PAR_MIN_FRAMES: usize = 512;

pub struct Frontend {
  mels: usize,
  window: Vec<f32>,
  /// Concatenated nonzero weights of every mel filter, in filter order.
  weights: Vec<f32>,
  /// Per filter, `(first_bin, len)` of its nonzero weights.
  spans: Vec<(u16, u16)>,
  fft: Rfft8,
}

impl Frontend {
  pub fn new(g: &Gguf, mels: usize) -> Result<Self> {
    if mels == 0 {
      bail!("gigaam frontend mel count must be nonzero");
    }
    let filters = g.tensor("frontend.mel_filterbank", &[BINS, mels])?.to_f32()?;
    let mut weights = Vec::new();
    let mut spans = Vec::with_capacity(mels);
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
    Ok(Self {
      mels,
      window: g.tensor("frontend.window", &[N_FFT])?.to_f32()?,
      weights,
      spans,
      fft: Rfft8::new(N_FFT),
    })
  }

  /// Mel bands per frame.
  pub fn mels(&self) -> usize {
    self.mels
  }

  /// Features of 16 kHz PCM, time-major `[frames][mels]`.
  pub fn compute(&self, pcm: &[f32]) -> Vec<f32> {
    if pcm.len() < N_FFT {
      return Vec::new();
    }
    let frames = (pcm.len() - N_FFT) / HOP + 1;
    let mels = self.mels;
    let mut out = vec![0f32; frames * mels];
    if frames >= PAR_MIN_FRAMES {
      let n_threads = crate::cpu::p_cores().clamp(1, 4);
      let per_frames = frames.div_ceil(n_threads).next_multiple_of(8);
      let per = per_frames * mels;
      let chunks: Vec<(usize, &mut [f32])> =
        out.chunks_mut(per).enumerate().map(|(i, s)| (i * per_frames, s)).collect();
      crate::cpu::par_each(
        chunks.into_iter(),
        || (),
        |_, (frame0, slice)| {
          self.compute_range(pcm, frame0, slice);
        },
      );
    } else {
      self.compute_range(pcm, 0, &mut out);
    }
    out
  }

  fn compute_range(&self, pcm: &[f32], frame0: usize, out: &mut [f32]) {
    let mels = self.mels;
    let (mut re, mut im) = ([[0f32; 8]; N_FFT / 2], [[0f32; 8]; N_FFT / 2]);
    let mut power = [[0f32; 8]; BINS];

    for (batch, chunk) in out.chunks_mut(8 * mels).enumerate() {
      let base = frame0 + batch * 8;
      let count = chunk.len() / mels;
      let last = count - 1;
      let fs = [
        &pcm[base * HOP..][..N_FFT],
        &pcm[(base + 1.min(last)) * HOP..][..N_FFT],
        &pcm[(base + 2.min(last)) * HOP..][..N_FFT],
        &pcm[(base + 3.min(last)) * HOP..][..N_FFT],
        &pcm[(base + 4.min(last)) * HOP..][..N_FFT],
        &pcm[(base + 5.min(last)) * HOP..][..N_FFT],
        &pcm[(base + 6.min(last)) * HOP..][..N_FFT],
        &pcm[(base + 7.min(last)) * HOP..][..N_FFT],
      ];
      self.fft.power_windowed(fs, &self.window, &mut re, &mut im, &mut power);
      ops::mel_filter8_f32(&self.weights, &self.spans, &power, chunk, count);
      ops::ln_clamp_inplace(chunk, 1e-9, 1e9);
    }
  }
}
