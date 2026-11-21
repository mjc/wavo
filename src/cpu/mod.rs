// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! CPU dot products for the frontends and decoders, with NEON tiles on aarch64 and AVX2 on x86_64,
//! and the encoders' kernels for the CPU backend.

pub mod conformer;
pub mod gemm;
pub mod ops;
pub mod transformer;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, OnceLock, PoisonError};

use half::f16;

use crate::error::{Result, bail};
use crate::gguf::{Tensor, Type, q4_codes, q4k_group_dm, q4k_packed, q5_codes};
use crate::{Api, Device, DeviceKind, Kernels};

/// Dot product with 16 independent accumulators (four FMA vectors in flight), summed pairwise so
/// the final reduction also vectorizes.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
  #[cfg(target_arch = "x86_64")]
  if x86::avx2() {
    // SAFETY: the CPU has AVX2 and FMA.
    return unsafe { x86::dot(a, b) };
  }

  #[cfg(target_arch = "aarch64")]
  {
    use std::arch::aarch64::*;
    let len = a.len().min(b.len());
    let n = len / 16 * 16;
    // SAFETY: NEON is part of aarch64 and every load reads 4 floats below `n` <= `len`.
    let head = unsafe {
      let (mut a0, mut a1, mut a2, mut a3) =
        (vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0), vdupq_n_f32(0.0));
      let (ap, bp) = (a.as_ptr(), b.as_ptr());
      let mut i = 0;
      while i < n {
        a0 = vfmaq_f32(a0, vld1q_f32(ap.add(i)), vld1q_f32(bp.add(i)));
        a1 = vfmaq_f32(a1, vld1q_f32(ap.add(i + 4)), vld1q_f32(bp.add(i + 4)));
        a2 = vfmaq_f32(a2, vld1q_f32(ap.add(i + 8)), vld1q_f32(bp.add(i + 8)));
        a3 = vfmaq_f32(a3, vld1q_f32(ap.add(i + 12)), vld1q_f32(bp.add(i + 12)));
        i += 16;
      }
      let lanes = vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3));
      let pair = vadd_f32(vget_low_f32(lanes), vget_high_f32(lanes));
      vget_lane_f32::<0>(pair) + vget_lane_f32::<1>(pair)
    };
    return head + a[n..len].iter().zip(&b[n..len]).map(|(x, y)| x * y).sum::<f32>();
  }

  #[allow(unreachable_code)]
  {
    let ((a16, a_tail), (b16, b_tail)) = (a.as_chunks::<16>(), b.as_chunks::<16>());
    let mut acc = [0f32; 16];
    for (x, y) in a16.iter().zip(b16) {
      for ((s, x), y) in acc.iter_mut().zip(x).zip(y) {
        *s = x.mul_add(*y, *s);
      }
    }

    for n in [8, 4, 2, 1] {
      let (lo, hi) = acc.split_at_mut(n);
      for (l, h) in lo.iter_mut().zip(&*hi) {
        *l += h;
      }
    }

    acc[0] + a_tail.iter().zip(b_tail).map(|(x, y)| x * y).sum::<f32>()
  }
}

/// The VNNI the encoders' GEMM takes, for `device`: `", AVX-VNNI"`, `", AVX512-VNNI"` or none.
fn vnni() -> &'static str {
  #[cfg(target_arch = "x86_64")]
  return match x86::vnni() {
    Some(x86::Vnni::Avx) => ", AVX-VNNI",
    Some(x86::Vnni::Avx512) => ", AVX512-VNNI",
    None => "",
  };
  #[allow(unreachable_code)]
  ""
}

/// The code path `tile` and `Q8::rows` take on this CPU: `NEON`, `AVX2+FMA` or `portable`.
pub fn path() -> &'static str {
  #[cfg(target_arch = "x86_64")]
  if x86::avx2() {
    return "AVX2+FMA";
  }

  if cfg!(target_arch = "aarch64") { "NEON" } else { "portable" }
}

/// What `Model::device` reports on the CPU: `CPU (AVX2+FMA, AVX-VNNI, 4 threads)`.
pub fn device() -> &'static Device {
  static DEVICE: OnceLock<Device> = OnceLock::new();
  DEVICE.get_or_init(|| Device {
    name: "CPU".into(),
    api: Api::Cpu,
    kind: DeviceKind::Cpu,
    driver: format!("{}{}, {} threads", path(), vnni(), threads()),
    kernels: Kernels::Cpu,
  })
}

/// Performance-core count on Apple Silicon (`hw.perflevel0.physicalcpu`), or `threads()` elsewhere.
pub fn p_cores() -> usize {
  static PCORES: OnceLock<usize> = OnceLock::new();
  *PCORES.get_or_init(|| {
    #[cfg(target_os = "macos")]
    {
      unsafe extern "C" {
        fn sysctlbyname(
          name: *const std::ffi::c_char,
          oldp: *mut std::ffi::c_void,
          oldlenp: *mut usize,
          newp: *const std::ffi::c_void,
          newlen: usize,
        ) -> i32;
      }
      let mut n: u32 = 0;
      let mut len = std::mem::size_of::<u32>();
      // SAFETY: `name` is NUL-terminated and `oldp`/`oldlenp` point to `n`.
      let rc = unsafe {
        sysctlbyname(
          c"hw.perflevel0.physicalcpu".as_ptr(),
          (&mut n as *mut u32).cast(),
          &mut len,
          std::ptr::null(),
          0,
        )
      };
      if rc == 0 && n > 0 {
        return n as usize;
      }
    }
    threads()
  })
}

static BENCH_P_CORES: AtomicBool = AtomicBool::new(false);

#[cfg(target_os = "macos")]
fn set_p_core_qos() {
  unsafe extern "C" {
    fn pthread_set_qos_class_self_np(qos_class: u32, relative_priority: i32) -> i32;
  }
  // QOS_CLASS_USER_INTERACTIVE (0x21).
  // SAFETY: libSystem FFI call with a valid constant QoS class.
  unsafe {
    let _ = pthread_set_qos_class_self_np(0x21, 0);
  }
}

/// Ask macOS to schedule the calling thread and newly spawned CPU pool workers on performance
/// cores; called only by `wavo bench` so benchmark timings aren't skewed by efficiency-core
/// placement.
pub fn bench_prefer_p_cores() {
  BENCH_P_CORES.store(true, Ordering::Relaxed);
  #[cfg(target_os = "macos")]
  set_p_core_qos();
}

/// Threads for the encoders' kernels: the CPUs this process may run on (`taskset` and cgroup
/// limits count).
pub fn threads() -> usize {
  static THREADS: OnceLock<usize> = OnceLock::new();
  *THREADS.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
}

#[derive(Clone, Copy)]
struct PoolJob {
  func: fn(*const ()),
  data: *const (),
  generation: usize,
}

// SAFETY: `run_threads` waits until every woken worker finishes before `data` can go out of scope.
unsafe impl Send for PoolJob {}

struct Worker {
  job: Mutex<PoolJob>,
  wake: Condvar,
}

struct Pool {
  workers: &'static [Worker],
  done: &'static AtomicUsize,
  busy: Mutex<()>,
}

fn pool() -> &'static Pool {
  static POOL: OnceLock<Pool> = OnceLock::new();
  POOL.get_or_init(|| {
    let n = threads().saturating_sub(1);
    let done: &'static AtomicUsize = Box::leak(Box::new(AtomicUsize::new(0)));
    let workers: &'static [Worker] = Box::leak(
      (0..n)
        .map(|_| Worker {
          job: Mutex::new(PoolJob { func: |_| {}, data: std::ptr::null(), generation: 0 }),
          wake: Condvar::new(),
        })
        .collect::<Vec<_>>()
        .into_boxed_slice(),
    );
    for (id, worker) in workers.iter().enumerate() {
      let _ = std::thread::Builder::new().name(format!("wavo-cpu-{id}")).spawn(move || {
        #[cfg(target_os = "macos")]
        if BENCH_P_CORES.load(Ordering::Relaxed) {
          set_p_core_qos();
        }
        let mut seen = 0usize;
        loop {
          let job = {
            let mut guard = worker.job.lock().unwrap_or_else(PoisonError::into_inner);
            while guard.generation == seen {
              guard = worker.wake.wait(guard).unwrap_or_else(PoisonError::into_inner);
            }
            seen = guard.generation;
            *guard
          };
          let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (job.func)(job.data)));
          done.fetch_add(1, Ordering::Release);
        }
      });
    }
    Pool { workers, done, busy: Mutex::new(()) }
  })
}

/// Runs `work(tid)` on `n` threads (`0..n`, with `0` on the calling thread), using the persistent
/// worker pool when uncontended and falling back to `std::thread::scope` if nested or busy.
pub fn run_threads<F: Fn(usize) + Sync>(n: usize, work: &F) {
  if n <= 1 {
    work(0);
    return;
  }
  let p = pool();
  let extra = (n - 1).min(p.workers.len());
  if extra > 0
    && let Ok(_busy) = p.busy.try_lock()
  {
    struct Runner<'a, F: Fn(usize) + Sync> {
      work: &'a F,
      next_id: AtomicUsize,
    }
    fn invoke<F: Fn(usize) + Sync>(ptr: *const ()) {
      // SAFETY: `ptr` points to `Runner<F>` alive until `WaitGuard::drop` completes.
      let r = unsafe { &*ptr.cast::<Runner<'_, F>>() };
      let tid = r.next_id.fetch_add(1, Ordering::Relaxed);
      (r.work)(tid);
    }
    struct WaitGuard<'a> {
      done: &'a AtomicUsize,
      expected: usize,
    }
    impl Drop for WaitGuard<'_> {
      fn drop(&mut self) {
        let mut spins = 0u32;
        while self.done.load(Ordering::Acquire) < self.expected {
          spins += 1;
          if spins < 256 {
            std::hint::spin_loop();
          } else {
            std::thread::yield_now();
          }
        }
      }
    }

    let runner = Runner { work, next_id: AtomicUsize::new(1) };
    let data = (&runner as *const Runner<'_, _>).cast();
    p.done.store(0, Ordering::Relaxed);
    for worker in &p.workers[..extra] {
      {
        let mut guard = worker.job.lock().unwrap_or_else(PoisonError::into_inner);
        guard.func = invoke::<F>;
        guard.data = data;
        guard.generation = guard.generation.wrapping_add(1);
      }
      worker.wake.notify_one();
    }
    let _wait = WaitGuard { done: p.done, expected: extra };
    work(0);
    return;
  }

  std::thread::scope(|s| {
    for tid in 1..n {
      s.spawn(move || work(tid));
    }
    work(0);
  });
}

/// `f(scratch, item)` for each item, on at most `max_threads` worker threads (no more than there
/// can be items) that take the next item as they come free, each with its own `scratch` from
/// `init`.
pub fn par_each_n<T: Send, S>(
  max_threads: usize,
  items: impl Iterator<Item = T> + Send,
  init: impl Fn() -> S + Sync,
  f: impl Fn(&mut S, T) + Sync,
) {
  let spawn = max_threads.max(1).min(items.size_hint().1.unwrap_or(usize::MAX));
  let items = Mutex::new(items);
  let next = || items.lock().unwrap_or_else(PoisonError::into_inner).next();
  let work = |_| {
    let Some(first) = next() else { return };
    let mut scratch = init();
    f(&mut scratch, first);
    while let Some(item) = next() {
      f(&mut scratch, item);
    }
  };
  run_threads(spawn, &work);
}

/// `f(scratch, item)` for each item, on `threads` worker threads (no more than there can be
/// items) that take the next item as they come free, each with its own `scratch` from `init`.
pub fn par_each<T: Send, S>(
  items: impl Iterator<Item = T> + Send,
  init: impl Fn() -> S + Sync,
  f: impl Fn(&mut S, T) + Sync,
) {
  par_each_n(threads(), items, init, f);
}

/// `w` (`k`-long rows) padded with zero rows to a multiple of 4, as `matmul` takes it.
pub fn rows4(mut w: Vec<f32>, k: usize) -> Vec<f32> {
  w.resize((w.len() / k).next_multiple_of(4) * k, 0.0);
  w
}

/// `[frame][row]` dots of the `k`-long rows of `w` (a multiple of 4 rows) with the `k`-long frames
/// of `z`, each equal to `dot(row, frame)`. Frames go in pairs, two rows at a time; a last single
/// frame goes four rows at a time. Either way each loaded chunk serves several dots.
pub fn matmul(w: &[f32], z: &[f32], k: usize) -> Vec<f32> {
  let rows = w.len() / k;
  let mut out = vec![0.0; z.len() / k * rows];
  let pairs = z.chunks_exact(2 * k);
  let last = pairs.remainder();
  for (z, out) in pairs.zip(out.chunks_exact_mut(2 * rows)) {
    for (r, w) in w.chunks_exact(2 * k).enumerate() {
      let [d0, d1] = tile::<2, 2>(w, z, k);
      out[2 * r..][..2].copy_from_slice(&d0);
      out[rows + 2 * r..][..2].copy_from_slice(&d1);
    }
  }

  if !last.is_empty() {
    let out = out.rchunks_exact_mut(rows).next().unwrap();
    for (w, out) in w.chunks_exact(4 * k).zip(out.as_chunks_mut::<4>().0) {
      [*out] = tile::<4, 1>(w, last, k);
    }
  }

  out
}

/// `dot` of `R` rows of `w` with `F` frames of `z` (`[frame][row]`), loading each chunk once. It
/// keeps `dot`'s lanes, fused multiply-adds and pairwise sum, so the results are identical.
#[cfg(target_arch = "aarch64")]
fn tile<const R: usize, const F: usize>(w: &[f32], z: &[f32], k: usize) -> [[f32; R]; F] {
  use std::arch::aarch64::*;
  let (w, z, n) = (&w[..R * k], &z[..F * k], k / 16 * 16);
  // SAFETY: NEON is part of aarch64, and every load reads 4 floats below `n` ≤ k of a row of `w`
  // or a frame of `z`.
  let acc = unsafe {
    let mut acc = [[[vdupq_n_f32(0.0); 4]; R]; F];
    for i in (0..n).step_by(16) {
      for c in 0..4 {
        let wv: [_; R] = std::array::from_fn(|r| vld1q_f32(w.as_ptr().add(r * k + i + 4 * c)));
        for (f, acc) in acc.iter_mut().enumerate() {
          let zv = vld1q_f32(z.as_ptr().add(f * k + i + 4 * c));
          for (a, wv) in acc.iter_mut().zip(wv) {
            a[c] = vfmaq_f32(a[c], wv, zv);
          }
        }
      }
    }

    acc.map(|acc| {
      acc.map(|[a0, a1, a2, a3]| {
        let lanes = vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3));
        let pair = vadd_f32(vget_low_f32(lanes), vget_high_f32(lanes));
        vget_lane_f32::<0>(pair) + vget_lane_f32::<1>(pair)
      })
    })
  };
  let mut out = acc;
  if n < k {
    for f in 0..F {
      let zt = &z[f * k + n..][..k - n];
      for r in 0..R {
        let wt = &w[r * k + n..][..k - n];
        out[f][r] += wt.iter().zip(zt).map(|(x, y)| x * y).sum::<f32>();
      }
    }
  }
  out
}

#[cfg(not(target_arch = "aarch64"))]
fn tile<const R: usize, const F: usize>(w: &[f32], z: &[f32], k: usize) -> [[f32; R]; F] {
  #[cfg(target_arch = "x86_64")]
  if x86::avx2() {
    // SAFETY: the CPU has AVX2 and FMA.
    return unsafe { x86::tile(w, z, k) };
  }

  std::array::from_fn(|f| std::array::from_fn(|r| dot(&w[r * k..][..k], &z[f * k..][..k])))
}

/// Weights `[rows][k]` for `matvecs`. Q8_0, Q5_0 and Q4_0 keep signed codes and one f32 scale per
/// 32 weights (`q · d`, exact in f32). Q4_K keeps nibbles and the f32 products `d·scale` and
/// `dmin·min` per 32 (`code · d · scale − dmin · min`, exact in f32). Q4_1 and Q5_1 expand to
/// f32, because each block also has a minimum; the flat length has to be a whole number of
/// blocks, or the last row is short. Q5_K expands to f32: each block has a superblock scale and
/// minimum. Q6_K expands to f32: each weight is a 6-bit code times an i8 scale times the
/// superblock scale. F32 and F16 stay as f32, as the reference reads them.
pub enum Matrix {
  Q8(Q8),
  Q4K(Q4K),
  F32 { k: usize, w: Vec<f32> },
}

impl Matrix {
  pub fn new(t: &Tensor) -> Result<Self> {
    let k = t.dims[0];
    if matches!(t.ty, Type::Q8_0 | Type::Q5_0 | Type::Q4_0) && !k.is_multiple_of(32) {
      bail!("quantized rows of {k} weights are not a multiple of 32");
    }
    // Q4_1 and Q5_1 decode whole blocks into a flat buffer. A short buffer panics on the last row.
    let n = t.len();
    if matches!(t.ty, Type::Q4_1 | Type::Q5_1) && !n.is_multiple_of(32) {
      bail!("quantized weights of {n} are not a whole number of blocks");
    }
    Ok(match t.ty {
      Type::F32 | Type::F16 | Type::Q4_1 | Type::Q5_1 | Type::Q5_K | Type::Q6_K => {
        Self::F32 { k, w: t.to_f32()? }
      }
      Type::Q4_K => Self::Q4K(Q4K::new(t)?),
      Type::Q8_0 => Self::Q8(Q8::new(&[t])?),
      Type::Q5_0 => {
        let mut b = Vec::new();
        t.read(1 << 20, |_, piece| b.extend_from_slice(piece))?;
        let blocks = b.as_chunks::<22>().0;
        let d = blocks.iter().map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect();
        let q = blocks.iter().flat_map(|b| q5_codes(b).map(|c| (c as i8) - 16)).collect();
        Self::Q8(Q8 { k, q, d })
      }
      Type::Q4_0 => {
        let mut b = Vec::new();
        t.read(1 << 20, |_, piece| b.extend_from_slice(piece))?;
        let blocks = b.as_chunks::<18>().0;
        let d = blocks.iter().map(|b| f16::from_le_bytes([b[0], b[1]]).to_f32()).collect();
        let q = blocks
          .iter()
          .flat_map(|b| {
            let qs: &[u8; 16] = b[2..].try_into().unwrap();
            q4_codes(qs).map(|c| (c as i8) - 8)
          })
          .collect();
        Self::Q8(Q8 { k, q, d })
      }
    })
  }

  pub fn row(&self, r: usize) -> Vec<f32> {
    match self {
      Self::Q8(q8) => q8.row(r),
      Self::Q4K(q4) => q4.row(r),
      Self::F32 { k, w } => w[r * k..][..*k].to_vec(),
    }
  }

  fn len(&self) -> usize {
    match self {
      Self::Q8(q8) => q8.q.len() / q8.k,
      Self::Q4K(q4) => q4.q.len() * 2 / q4.k,
      Self::F32 { k, w } => w.len() / k,
    }
  }

  /// Rows from `r0` times `x` into `out`.
  fn rows(&self, r0: usize, x: &[f32], out: &mut [f32]) {
    match self {
      Self::Q8(q8) => q8.rows(r0, x, out),
      Self::Q4K(q4) => q4.rows(r0, x, out),
      Self::F32 { k, w } => {
        for (o, w) in out.iter_mut().zip(w[r0 * k..].chunks_exact(*k)) {
          *o = dot(w, x);
        }
      }
    }
  }
}

/// Q4_K weights `[rows][k]`. `q` holds two nibbles per byte in weight order (low nibble first).
/// `d` and `m` are `d·scale` and `dmin·min` for each flat group of 32. A weight is
/// `code · d − m`, the same f32 as [`Tensor::to_f32`](crate::gguf::Tensor::to_f32).
pub(crate) struct Q4K {
  k: usize,
  q: Vec<u8>,
  d: Vec<f32>,
  m: Vec<f32>,
}

impl Q4K {
  fn new(t: &Tensor) -> Result<Self> {
    let n = t.len();
    if !n.is_multiple_of(256) {
      bail!("Q4_K weights of {n} are not a whole number of blocks");
    }
    let (mut q, mut d, mut m) =
      (Vec::with_capacity(n / 2), Vec::with_capacity(n / 32), Vec::with_capacity(n / 32));
    // Pieces of whole 144-byte blocks.
    t.read(144 << 13, |_, piece| {
      for block in piece.as_chunks::<144>().0 {
        q.extend_from_slice(&q4k_packed(block));
        for (dj, mj) in q4k_group_dm(block) {
          d.push(dj);
          m.push(mj);
        }
      }
    })?;
    Ok(Self { k: t.dims[0], q, d, m })
  }

  fn code(&self, flat: usize) -> f32 {
    let byte = self.q[flat / 2];
    let code = if flat.is_multiple_of(2) { byte & 15 } else { byte >> 4 };
    self.d[flat / 32] * code as f32 - self.m[flat / 32]
  }

  fn row(&self, r: usize) -> Vec<f32> {
    let base = r * self.k;
    (0..self.k).map(|i| self.code(base + i)).collect()
  }

  /// Rows from `r0` times `x` into `out`. A row whose width is a multiple of 32 dequantizes in
  /// the AVX2 tile; every other width dequantizes one row and calls [`dot`].
  fn rows(&self, r0: usize, x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if x86::avx2() && self.k.is_multiple_of(32) {
      // SAFETY: the CPU has AVX2 and FMA, and each row is a whole number of groups of 32.
      return unsafe { x86::q4k_rows(self, r0, x, out) };
    }

    let mut row = vec![0.0; self.k];
    for (i, o) in out.iter_mut().enumerate() {
      let base = (r0 + i) * self.k;
      for (c, w) in row.iter_mut().enumerate() {
        *w = self.code(base + c);
      }
      *o = dot(&row, x);
    }
  }
}

/// Q8_0, Q5_0 and Q4_0 weights `[rows][k]` (k a multiple of 32) as int8 codes and one f32 scale
/// per 32. A Q8_0 code is the file's i8; a Q5_0 code is `code - 16` (−16..=15); a Q4_0 code is
/// `code - 8` (−8..=7). A weight is `q · d`, exact in f32, so every product below equals `dot` on
/// the dequantized row.
pub struct Q8 {
  k: usize,
  q: Vec<i8>,
  d: Vec<f32>,
}

impl Q8 {
  /// The Q8_0 tensors `ts`, of rows of the same length, stacked.
  pub fn new(ts: &[&Tensor]) -> Result<Self> {
    let len = ts.iter().map(|t| t.len()).sum::<usize>();
    let (mut q, mut d) = (Vec::with_capacity(len), Vec::with_capacity(len / 32));
    for t in ts {
      // Pieces of whole 34-byte blocks.
      t.read(34 << 15, |_, piece| {
        for b in piece.as_chunks::<34>().0 {
          d.push(f16::from_le_bytes([b[0], b[1]]).to_f32());
          q.extend(b[2..].iter().map(|&v| v as i8));
        }
      })?;
    }

    Ok(Self { k: ts[0].dims[0], q, d })
  }

  fn row(&self, r: usize) -> Vec<f32> {
    let (q, d) = (&self.q[r * self.k..][..self.k], &self.d[r * self.k / 32..]);
    q.iter().enumerate().map(|(i, &q)| q as f32 * d[i / 32]).collect()
  }

  /// Rows from `r0` times `x` into `out`, four rows per pass over x, stepping by full 32-weight
  /// Q8 blocks so each scale is loaded once with no inner bounds checks.
  #[cfg(target_arch = "aarch64")]
  fn rows(&self, r0: usize, x: &[f32], out: &mut [f32]) {
    use std::arch::aarch64::*;
    let k = self.k;
    let blocks = k / 32;
    let (q, d, x) =
      (&self.q[r0 * k..][..out.len() * k], &self.d[r0 * blocks..][..out.len() * blocks], &x[..k]);
    // SAFETY: NEON is part of aarch64; `k` is a multiple of 32 (`blocks = k / 32`), so every
    // 32-byte load of `q`, 32-float load of `x`, and scale load of `d` stays in bounds; a short
    // last group repeats its last row.
    unsafe {
      for (g, out) in out.chunks_mut(4).enumerate() {
        let last = out.len() - 1;
        let rows = [4 * g, 4 * g + 1.min(last), 4 * g + 2.min(last), 4 * g + 3.min(last)];
        let q_ptrs = [
          q.as_ptr().add(rows[0] * k),
          q.as_ptr().add(rows[1] * k),
          q.as_ptr().add(rows[2] * k),
          q.as_ptr().add(rows[3] * k),
        ];
        let d_ptrs = [
          d.as_ptr().add(rows[0] * blocks),
          d.as_ptr().add(rows[1] * blocks),
          d.as_ptr().add(rows[2] * blocks),
          d.as_ptr().add(rows[3] * blocks),
        ];
        let mut acc = [[vdupq_n_f32(0.0); 4]; 4];
        for blk in 0..blocks {
          let xp = x.as_ptr().add(blk * 32);
          let xs0 =
            [vld1q_f32(xp), vld1q_f32(xp.add(4)), vld1q_f32(xp.add(8)), vld1q_f32(xp.add(12))];
          let xs1 = [
            vld1q_f32(xp.add(16)),
            vld1q_f32(xp.add(20)),
            vld1q_f32(xp.add(24)),
            vld1q_f32(xp.add(28)),
          ];
          for j in 0..4 {
            let scale = *d_ptrs[j].add(blk);
            let qp = q_ptrs[j].add(blk * 32);
            let v0 = vld1q_s8(qp);
            let (lo0, hi0) = (vmovl_s8(vget_low_s8(v0)), vmovl_high_s8(v0));
            let w0 = [vget_low_s16(lo0), vget_high_s16(lo0), vget_low_s16(hi0), vget_high_s16(hi0)];
            for c in 0..4 {
              let w = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(w0[c])), scale);
              acc[j][c] = vfmaq_f32(acc[j][c], w, xs0[c]);
            }
            let v1 = vld1q_s8(qp.add(16));
            let (lo1, hi1) = (vmovl_s8(vget_low_s8(v1)), vmovl_high_s8(v1));
            let w1 = [vget_low_s16(lo1), vget_high_s16(lo1), vget_low_s16(hi1), vget_high_s16(hi1)];
            for c in 0..4 {
              let w = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(w1[c])), scale);
              acc[j][c] = vfmaq_f32(acc[j][c], w, xs1[c]);
            }
          }
        }

        for (o, [a0, a1, a2, a3]) in out.iter_mut().zip(acc) {
          let lanes = vaddq_f32(vaddq_f32(a0, a2), vaddq_f32(a1, a3));
          let pair = vadd_f32(vget_low_f32(lanes), vget_high_f32(lanes));
          *o = vget_lane_f32::<0>(pair) + vget_lane_f32::<1>(pair);
        }
      }
    }
  }

  #[cfg(not(target_arch = "aarch64"))]
  fn rows(&self, r0: usize, x: &[f32], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if x86::avx2() {
      // SAFETY: the CPU has AVX2 and FMA.
      return unsafe { x86::rows(self, r0, x, out) };
    }

    for (r, o) in out.iter_mut().enumerate() {
      *o = dot(&self.row(r0 + r), x);
    }
  }
}

/// `tile` and `Q8::rows` with AVX2 and FMA, chosen at run time. Like the NEON ones they keep
/// `dot`'s 16 lanes (two 8-float vectors), fused multiply-adds and pairwise sum, so the results
/// are identical. No intrinsics in closures: those would not get the target features.
#[cfg(target_arch = "x86_64")]
mod x86 {
  use std::arch::x86_64::*;

  use super::{OnceLock, Q4K, Q8};

  pub fn avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
  }

  /// The `vpdpwssd` of the encoders' i16 GEMM tile on 256-bit vectors.
  #[derive(Clone, Copy, PartialEq, Eq)]
  pub enum Vnni {
    /// AVX-VNNI (VEX): Intel since Alder Lake, AMD since Zen 5.
    Avx,
    /// AVX512-VNNI with AVX512VL (EVEX): Intel's servers since Cascade Lake, AMD's Zen 4.
    Avx512,
  }

  /// The VNNI on this CPU with AVX2 and FMA, unless `WAVO_KERNELS=portable` asks for the f32 tile.
  pub fn vnni() -> Option<Vnni> {
    static VNNI: OnceLock<Option<Vnni>> = OnceLock::new();
    *VNNI.get_or_init(|| {
      if !avx2() || std::env::var_os("WAVO_KERNELS").is_some_and(|v| v == "portable") {
        return None;
      }

      if is_x86_feature_detected!("avxvnni") {
        return Some(Vnni::Avx);
      }

      let avx512 = is_x86_feature_detected!("avx512vnni") && is_x86_feature_detected!("avx512vl");
      avx512.then_some(Vnni::Avx512)
    })
  }

  /// `dot`'s pairwise sum of 16 lanes, 0..8 in `a` and 8..16 in `b`.
  #[target_feature(enable = "avx2,fma")]
  fn sum(a: __m256, b: __m256) -> f32 {
    let s = _mm256_add_ps(a, b);
    let s = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps::<1>(s));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    _mm_cvtss_f32(s) + _mm_cvtss_f32(_mm_shuffle_ps::<1>(s, s))
  }

  /// `dot` without `mul_add`'s libm call.
  #[target_feature(enable = "avx2,fma")]
  pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) / 16 * 16;
    let mut acc = [_mm256_setzero_ps(); 2];
    for i in (0..n).step_by(16) {
      for (c, acc) in acc.iter_mut().enumerate() {
        // SAFETY: 8 floats below `n` of `a` and of `b`.
        let (x, y) = unsafe {
          (_mm256_loadu_ps(a.as_ptr().add(i + 8 * c)), _mm256_loadu_ps(b.as_ptr().add(i + 8 * c)))
        };
        *acc = _mm256_fmadd_ps(x, y, *acc);
      }
    }

    sum(acc[0], acc[1]) + a[n..].iter().zip(&b[n..]).map(|(x, y)| x * y).sum::<f32>()
  }

  #[target_feature(enable = "avx2,fma")]
  pub fn tile<const R: usize, const F: usize>(w: &[f32], z: &[f32], k: usize) -> [[f32; R]; F] {
    let (w, z, n) = (&w[..R * k], &z[..F * k], k / 16 * 16);
    let mut acc = [[[_mm256_setzero_ps(); 2]; R]; F];
    for i in (0..n).step_by(16) {
      for c in 0..2 {
        let mut wv = [_mm256_setzero_ps(); R];
        for (r, v) in wv.iter_mut().enumerate() {
          // SAFETY: 8 floats below `n` ≤ k of a row of `w`.
          *v = unsafe { _mm256_loadu_ps(w.as_ptr().add(r * k + i + 8 * c)) };
        }

        for (f, acc) in acc.iter_mut().enumerate() {
          // SAFETY: 8 floats below `n` ≤ k of a frame of `z`.
          let zv = unsafe { _mm256_loadu_ps(z.as_ptr().add(f * k + i + 8 * c)) };
          for (a, wv) in acc.iter_mut().zip(wv) {
            a[c] = _mm256_fmadd_ps(wv, zv, a[c]);
          }
        }
      }
    }

    let mut out = [[0.0; R]; F];
    for (f, out) in out.iter_mut().enumerate() {
      for (r, out) in out.iter_mut().enumerate() {
        let (w, z) = (&w[r * k + n..][..k - n], &z[f * k + n..][..k - n]);
        let tail = w.iter().zip(z).map(|(x, y)| x * y).sum::<f32>();
        *out = sum(acc[f][r][0], acc[f][r][1]) + tail;
      }
    }

    out
  }

  #[target_feature(enable = "avx2,fma")]
  pub fn rows(q8: &Q8, r0: usize, x: &[f32], out: &mut [f32]) {
    let k = q8.k;
    let (q, d, x) = (&q8.q[r0 * k..][..out.len() * k], &q8.d[r0 * k / 32..], &x[..k]);
    for (g, out) in out.chunks_mut(4).enumerate() {
      let mut acc = [[_mm256_setzero_ps(); 2]; 4];
      for i in (0..k).step_by(16) {
        // SAFETY: 16 floats of `x` below k.
        let xs =
          unsafe { [_mm256_loadu_ps(x.as_ptr().add(i)), _mm256_loadu_ps(x.as_ptr().add(i + 8))] };
        for (j, acc) in acc.iter_mut().enumerate() {
          // A short last group repeats its last row.
          let at = (4 * g + j.min(out.len() - 1)) * k + i;
          // SAFETY: 16 int8 of a row of `q` below k.
          let v = unsafe { _mm_loadu_si128(q.as_ptr().add(at).cast()) };
          let scale = _mm256_set1_ps(d[at / 32]);
          let lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
          let hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(v)));
          acc[0] = _mm256_fmadd_ps(_mm256_mul_ps(lo, scale), xs[0], acc[0]);
          acc[1] = _mm256_fmadd_ps(_mm256_mul_ps(hi, scale), xs[1], acc[1]);
        }
      }

      for (o, [a, b]) in out.iter_mut().zip(acc) {
        *o = sum(a, b);
      }
    }
  }

  /// `dot` of Q4_K rows: each group of 32 is `code · d − m` in registers, then the same 16-wide
  /// FMAs as [`dot`](super::dot). A short last group of rows repeats its last row.
  #[target_feature(enable = "avx2,fma")]
  pub fn q4k_rows(q4: &Q4K, r0: usize, x: &[f32], out: &mut [f32]) {
    let k = q4.k;
    let (q, d, m, x) = (&q4.q, &q4.d, &q4.m, &x[..k]);
    let nib = _mm_set1_epi8(0x0f);
    for (g, out) in out.chunks_mut(4).enumerate() {
      let mut acc = [[_mm256_setzero_ps(); 2]; 4];
      for i in (0..k).step_by(32) {
        // SAFETY: 32 floats of `x` below k.
        let xs = unsafe {
          [
            _mm256_loadu_ps(x.as_ptr().add(i)),
            _mm256_loadu_ps(x.as_ptr().add(i + 8)),
            _mm256_loadu_ps(x.as_ptr().add(i + 16)),
            _mm256_loadu_ps(x.as_ptr().add(i + 24)),
          ]
        };
        for (j, acc) in acc.iter_mut().enumerate() {
          let flat = (r0 + 4 * g + j.min(out.len() - 1)) * k + i;
          let scale = _mm256_set1_ps(d[flat / 32]);
          let min = _mm256_set1_ps(m[flat / 32]);
          for (half, xs) in xs.as_chunks::<2>().0.iter().enumerate() {
            let at = flat + 16 * half;
            // SAFETY: eight nibble bytes, weights `at..at + 16`, inside `q`. `movq` has no
            // alignment requirement.
            let raw = unsafe { _mm_loadl_epi64(q.as_ptr().add(at / 2).cast()) };
            let lo = _mm_and_si128(raw, nib);
            let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), nib);
            let seq = _mm_unpacklo_epi8(lo, hi);
            let c0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(seq));
            let c1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128::<8>(seq)));
            let w0 = _mm256_sub_ps(_mm256_mul_ps(c0, scale), min);
            let w1 = _mm256_sub_ps(_mm256_mul_ps(c1, scale), min);
            acc[0] = _mm256_fmadd_ps(w0, xs[0], acc[0]);
            acc[1] = _mm256_fmadd_ps(w1, xs[1], acc[1]);
          }
        }
      }

      for (o, [a, b]) in out.iter_mut().zip(acc) {
        *o = sum(a, b);
      }
    }
  }
}

/// `w · x` for each job `(w, x)`. Blocks of 64 rows go to four scoped threads (an M1/M2 has four
/// performance cores) as they come free, so a thread on an efficiency core takes fewer.
pub fn matvecs<const N: usize>(jobs: [(&Matrix, &[f32]); N]) -> [Vec<f32>; N] {
  let mut out = jobs.map(|(w, _)| vec![0.0; w.len()]);
  let blocks = out.iter_mut().zip(jobs).flat_map(|(out, (w, x))| {
    out.chunks_mut(64).enumerate().map(move |(i, out)| (w, x, 64 * i, out))
  });
  let blocks = Mutex::new(blocks);
  let next = || blocks.lock().unwrap_or_else(PoisonError::into_inner).next();
  let work = |_| {
    while let Some((w, x, r0, out)) = next() {
      w.rows(r0, x, out);
    }
  };
  run_threads(p_cores().min(4), &work);
  out
}

#[cfg(test)]
mod tests {
  use half::f16;

  use crate::gguf::{Gguf, Type};

  #[test]
  fn quantized_matrices_require_block_aligned_rows() {
    for (ty, block_size) in [(Type::Q8_0, 34), (Type::Q5_0, 22), (Type::Q4_0, 18)] {
      let bytes = vec![0; block_size];
      let g = Gguf::with_tensors(&[("w", &[40], ty, &bytes)]);
      assert!(super::Matrix::new(&g.tensor("w", &[40]).unwrap()).is_err(), "{ty:?}");

      let bytes = vec![0; block_size];
      let g = Gguf::with_tensors(&[("w", &[32], ty, &bytes)]);
      assert!(super::Matrix::new(&g.tensor("w", &[32]).unwrap()).is_ok(), "{ty:?}");
    }
  }

  #[test]
  fn affine_expansion_needs_whole_blocks() {
    // [40, 2] stores two blocks (64 values). The second 40-wide row used to panic.
    // Four rows of 40 are 160 values, five whole blocks, so a 40-wide row still loads.
    for (ty, block) in [(Type::Q4_1, 20usize), (Type::Q5_1, 24)] {
      let short = vec![0; 2 * block];
      let g = Gguf::with_tensors(&[("w", &[40, 2], ty, &short)]);
      match super::Matrix::new(&g.tensor("w", &[40, 2]).unwrap()) {
        Err(err) => {
          assert!(err.to_string().contains("not a whole number of blocks"), "{ty:?}: {err}")
        }
        Ok(_) => panic!("{ty:?} accepted a short expansion"),
      }

      let full = vec![0; 5 * block];
      let g = Gguf::with_tensors(&[("w", &[40, 4], ty, &full)]);
      let w = super::Matrix::new(&g.tensor("w", &[40, 4]).unwrap()).unwrap();
      assert_eq!(w.row(3), vec![0.0; 40], "{ty:?}");
    }
  }

  #[test]
  fn matmul_matches_dot() {
    let (k, rows, frames) = (333, 8, 5);
    let v: Vec<f32> =
      (0..(rows + frames) * k).map(|i| (i * 7919 % 1000) as f32 / 500.0 - 1.0).collect();
    let (z, w) = v.split_at(frames * k);
    let want: Vec<u32> = (z.chunks_exact(k))
      .flat_map(|z| w.chunks_exact(k).map(|r| super::dot(r, z).to_bits()))
      .collect();
    let got: Vec<u32> = super::matmul(w, z, k).into_iter().map(f32::to_bits).collect();
    assert_eq!(got, want);
  }

  #[test]
  fn matvecs_match_dot() {
    // 70 rows: a full 64-row block and a short one whose last 4-row group has 2 rows.
    let (k, rows) = (96, 70);
    let q = (0..rows * k).map(|i| (i * 7919 % 255) as u8 as i8).collect();
    let d = (0..rows * k / 32).map(|i| (i % 7 + 1) as f32 * 1e-3).collect();
    let w = super::Matrix::Q8(super::Q8 { k, q, d });
    let x: Vec<f32> = (0..k).map(|i| (i * 31 % 17) as f32 / 8.0 - 1.0).collect();
    let want: Vec<u32> = (0..rows).map(|r| super::dot(&w.row(r), &x).to_bits()).collect();
    let [got, again] = super::matvecs([(&w, &x), (&w, &x)]);
    assert_eq!(got.into_iter().map(f32::to_bits).collect::<Vec<_>>(), want);
    assert_eq!(again.into_iter().map(f32::to_bits).collect::<Vec<_>>(), want);
  }

  #[test]
  fn matrices_load_every_tensor_type() {
    // 70 rows of 64 weights q/8 with q in -15..=15: exact in all three types, Q8_0 with scale 1/8.
    let (k, rows) = (64, 70);
    let v: Vec<f32> = (0..rows * k)
      .map(|i| {
        let d = 0.5 * (1 + ((i / 32) % 3)) as f32;
        ((i % 31) as f32 - 15.0) * d
      })
      .collect();
    let f32s: Vec<u8> = v.iter().flat_map(|v| v.to_le_bytes()).collect();
    let f16s: Vec<u8> = v.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect();
    let mut q8 = Vec::new();
    for (i, b) in v.as_chunks::<32>().0.iter().enumerate() {
      let d = 0.5 * (1 + (i % 3)) as f32;
      q8.extend(f16::from_f32(d).to_le_bytes());
      q8.extend(b.map(|v| (v / d) as i8 as u8));
    }

    let x: Vec<f32> = (0..k).map(|i| (i * 31 % 17) as f32 / 8.0 - 1.0).collect();
    let want: Vec<u32> = v.chunks_exact(k).map(|r| super::dot(r, &x).to_bits()).collect();
    for (ty, bytes) in [(Type::F32, f32s), (Type::F16, f16s), (Type::Q8_0, q8)] {
      let g = Gguf::with_tensors(&[("w", &[k, rows], ty, &bytes)]);
      let w = super::Matrix::new(&g.tensor("w", &[k, rows]).unwrap()).unwrap();
      for r in 0..rows {
        assert_eq!(w.row(r), v[r * k..(r + 1) * k], "{ty:?}, row {r}");
      }
      let [got] = super::matvecs([(&w, &x)]);
      assert_eq!(got.into_iter().map(f32::to_bits).collect::<Vec<_>>(), want, "{ty:?}");
    }
  }

  #[test]
  fn q5_0_matches_dot() {
    // Each block is codes 0..31 at scale 0.5, which quantizes back to itself. 70 rows: a full
    // 64-row block and a short one whose last 4-row group has 2 rows.
    let (k, rows) = (64, 70);
    let v: Vec<f32> = (0..rows * k)
      .map(|i| {
        let d = 0.5 * (1 + ((i / 32) % 3)) as f32;
        ((i % 32) as f32 - 16.0) * d
      })
      .collect();
    let q5 = crate::gguf::quantize_q5_0(&v);
    let x: Vec<f32> = (0..k).map(|i| (i * 31 % 17) as f32 / 8.0 - 1.0).collect();
    let want: Vec<u32> = v.chunks_exact(k).map(|r| super::dot(r, &x).to_bits()).collect();
    let g = Gguf::with_tensors(&[("w", &[k, rows], Type::Q5_0, &q5)]);
    let w = super::Matrix::new(&g.tensor("w", &[k, rows]).unwrap()).unwrap();
    for r in 0..rows {
      assert_eq!(w.row(r), v[r * k..(r + 1) * k], "row {r}");
    }
    let [got] = super::matvecs([(&w, &x)]);
    assert_eq!(got.into_iter().map(f32::to_bits).collect::<Vec<_>>(), want);
  }

  #[test]
  fn block32_matches_dot() {
    // Grids that quantize back to themselves. Q4_0 stays on the Q8 tile; Q4_1 and Q5_1 expand.
    // Q4_K stays packed; its block is 256, so it has its own test.
    // 70 rows: a full 64-row block and a short one whose last 4-row group has 2 rows.
    let (k, rows) = (64, 70);
    let x: Vec<f32> = (0..k).map(|i| (i * 31 % 17) as f32 / 8.0 - 1.0).collect();
    for (ty, levels, offset) in
      [(Type::Q4_0, 16, -8.0), (Type::Q4_1, 16, -1.0), (Type::Q5_1, 32, 0.0)]
    {
      let v: Vec<f32> = (0..rows * k)
        .map(|i| {
          let scale = 0.5 * (1 + ((i / 32) % 3)) as f32;
          let code = (i % levels) as f32;
          if ty == Type::Q4_0 { (code - 8.0) * scale } else { code * scale + offset }
        })
        .collect();
      let bytes = match ty {
        Type::Q4_0 => crate::gguf::quantize_q4_0(&v),
        Type::Q4_1 => crate::gguf::quantize_q4_1(&v),
        Type::Q5_1 => crate::gguf::quantize_q5_1(&v),
        _ => unreachable!(),
      };
      let want: Vec<u32> = v.chunks_exact(k).map(|r| super::dot(r, &x).to_bits()).collect();
      let g = Gguf::with_tensors(&[("w", &[k, rows], ty, &bytes)]);
      let w = super::Matrix::new(&g.tensor("w", &[k, rows]).unwrap()).unwrap();
      for r in 0..rows {
        assert_eq!(w.row(r), v[r * k..(r + 1) * k], "{ty:?}, row {r}");
      }
      let [got] = super::matvecs([(&w, &x)]);
      assert_eq!(got.into_iter().map(f32::to_bits).collect::<Vec<_>>(), want, "{ty:?}");
    }
  }

  #[test]
  fn q4_k_matches_dot() {
    // The packed row matches the dequantized block, and the matvec matches a dot of that row.
    // 256×6 hits a short last group of rows. 320×4 crosses a 256-weight block inside the row.
    for (k, rows) in [(256, 6), (320, 4)] {
      let v: Vec<f32> = (0..rows * k).map(|i| (i % 19) as f32 / 3.0 - 2.0).collect();
      let bytes = crate::gguf::quantize_q4_k(&v);
      let x: Vec<f32> = (0..k).map(|i| (i * 31 % 17) as f32 / 8.0 - 1.0).collect();
      let g = Gguf::with_tensors(&[("w", &[k, rows], Type::Q4_K, &bytes)]);
      let t = g.tensor("w", &[k, rows]).unwrap();
      let deq = t.to_f32().unwrap();
      let w = super::Matrix::new(&t).unwrap();
      for r in 0..rows {
        assert_eq!(w.row(r), deq[r * k..(r + 1) * k], "k {k}, row {r}");
      }
      let want: Vec<u32> = deq.chunks_exact(k).map(|r| super::dot(r, &x).to_bits()).collect();
      let [got] = super::matvecs([(&w, &x)]);
      assert_eq!(got.into_iter().map(f32::to_bits).collect::<Vec<_>>(), want, "k {k}");
    }
  }
}
