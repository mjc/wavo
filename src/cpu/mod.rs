// Copyright (c) vladkens | MIT License | https://github.com/vladkens/wavo
//! CPU dot products for the frontends and decoders, with NEON tiles on aarch64 and AVX2 on x86_64,
//! and the encoders' kernels for the CPU backend.

pub mod conformer;
pub mod gemm;
pub mod ops;
pub mod transformer;

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError, TryLockError};

use half::f16;

use crate::error::Result;
use crate::gguf::{Tensor, Type};
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
        *s = ops::fma(*x, *y, *s);
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
  expected: usize,
}

// SAFETY: `run_threads` waits until every woken worker finishes before `data` can go out of scope.
unsafe impl Send for PoolJob {}

struct Worker {
  job: Mutex<PoolJob>,
  wake: Condvar,
}

struct Done {
  count: AtomicUsize,
  waiting: AtomicBool,
  lock: Mutex<()>,
  ready: Condvar,
}

struct Pool {
  workers: Mutex<Vec<Arc<Worker>>>,
  done: Arc<Done>,
}

fn spawn_worker(id: usize, done: Arc<Done>) -> Option<Arc<Worker>> {
  let worker = Arc::new(Worker {
    job: Mutex::new(PoolJob { func: |_| {}, data: std::ptr::null(), generation: 0, expected: 0 }),
    wake: Condvar::new(),
  });
  let w = Arc::clone(&worker);
  std::thread::Builder::new()
    .name(format!("wavo-cpu-{id}"))
    .spawn(move || {
      let mut seen = 0usize;
      loop {
        let job = {
          let mut guard = w.job.lock().unwrap_or_else(PoisonError::into_inner);
          while guard.generation == seen {
            guard = w.wake.wait(guard).unwrap_or_else(PoisonError::into_inner);
          }
          seen = guard.generation;
          *guard
        };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (job.func)(job.data)));
        let finished = done.count.fetch_add(1, Ordering::SeqCst) + 1;
        if finished == job.expected && done.waiting.load(Ordering::SeqCst) {
          let _guard = done.lock.lock().unwrap_or_else(PoisonError::into_inner);
          done.ready.notify_one();
        }
      }
    })
    .ok()?;
  Some(worker)
}

impl Pool {
  fn new() -> Self {
    Self {
      workers: Mutex::new(Vec::new()),
      done: Arc::new(Done {
        count: AtomicUsize::new(0),
        waiting: AtomicBool::new(false),
        lock: Mutex::new(()),
        ready: Condvar::new(),
      }),
    }
  }

  fn run_threads<F: Fn(usize) + Sync>(&self, n: usize, work: &F) {
    if n <= 1 {
      work(0);
      return;
    }
    let extra = n - 1;
    if extra <= threads().saturating_sub(1)
      && let Some(mut workers) = match self.workers.try_lock() {
        Ok(guard) => Some(guard),
        Err(TryLockError::Poisoned(e)) => Some(e.into_inner()),
        Err(TryLockError::WouldBlock) => None,
      }
    {
      while workers.len() < extra {
        let Some(worker) = spawn_worker(workers.len(), Arc::clone(&self.done)) else { break };
        workers.push(worker);
      }
      if extra <= workers.len() {
        struct Runner<'a, F: Fn(usize) + Sync> {
          work: &'a F,
          next_id: AtomicUsize,
          panic: Mutex<Option<Box<dyn std::any::Any + Send>>>,
        }
        fn invoke<F: Fn(usize) + Sync>(ptr: *const ()) {
          // SAFETY: `ptr` points to `Runner<F>` alive until `WaitGuard::drop` completes.
          let r = unsafe { &*ptr.cast::<Runner<'_, F>>() };
          let tid = r.next_id.fetch_add(1, Ordering::Relaxed);
          if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (r.work)(tid))) {
            let mut slot = r.panic.lock().unwrap_or_else(PoisonError::into_inner);
            if slot.is_none() {
              *slot = Some(e);
            }
          }
        }
        struct WaitGuard<'a> {
          done: &'a Done,
          expected: usize,
        }
        impl Drop for WaitGuard<'_> {
          fn drop(&mut self) {
            for _ in 0..256 {
              if self.done.count.load(Ordering::Acquire) >= self.expected {
                return;
              }
              std::hint::spin_loop();
            }
            let mut guard = self.done.lock.lock().unwrap_or_else(PoisonError::into_inner);
            self.done.waiting.store(true, Ordering::SeqCst);
            while self.done.count.load(Ordering::SeqCst) < self.expected {
              guard = self.done.ready.wait(guard).unwrap_or_else(PoisonError::into_inner);
            }
          }
        }

        let runner = Runner { work, next_id: AtomicUsize::new(1), panic: Mutex::new(None) };
        let data = (&runner as *const Runner<'_, _>).cast();
        self.done.count.store(0, Ordering::Relaxed);
        self.done.waiting.store(false, Ordering::Relaxed);
        for worker in &workers[..extra] {
          {
            let mut guard = worker.job.lock().unwrap_or_else(PoisonError::into_inner);
            guard.func = invoke::<F>;
            guard.data = data;
            guard.generation = guard.generation.wrapping_add(1);
            guard.expected = extra;
          }
          worker.wake.notify_one();
        }
        let wait = WaitGuard { done: &self.done, expected: extra };
        work(0);
        drop(wait);
        drop(workers);
        if let Some(e) = runner.panic.into_inner().unwrap_or_else(PoisonError::into_inner) {
          std::panic::resume_unwind(e);
        }
        return;
      }
      drop(workers);
    }

    std::thread::scope(|s| {
      for tid in 1..n {
        s.spawn(move || work(tid));
      }
      work(0);
    });
  }
}

fn pool() -> &'static Pool {
  static POOL: OnceLock<Pool> = OnceLock::new();
  POOL.get_or_init(Pool::new)
}

/// Runs `work(tid)` on `n` threads (`0..n`, with `0` on the calling thread), using the persistent
/// worker pool when uncontended (growing lazily up to `threads() - 1` workers) and falling back to
/// `std::thread::scope` if nested, busy, or a worker fails to spawn.
pub fn run_threads<F: Fn(usize) + Sync>(n: usize, work: &F) {
  pool().run_threads(n, work);
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

/// Weights `[rows][k]` for `matvecs`, kept as Q8_0 or, from an F32 or F16 tensor, as f32 (exact,
/// as the reference reads them).
pub enum Matrix {
  Q8(Q8),
  F32 { k: usize, w: Vec<f32> },
}

impl Matrix {
  pub fn new(t: &Tensor) -> Result<Self> {
    let k = t.dims[0];
    Ok(match t.ty {
      Type::F32 | Type::F16 => Self::F32 { k, w: t.to_f32()? },
      Type::Q8_0 => Self::Q8(Q8::new(&[t])?),
    })
  }

  pub fn row(&self, r: usize) -> Vec<f32> {
    match self {
      Self::Q8(q8) => q8.row(r),
      Self::F32 { k, w } => w[r * k..][..*k].to_vec(),
    }
  }

  fn len(&self) -> usize {
    match self {
      Self::Q8(q8) => q8.q.len() / q8.k,
      Self::F32 { k, w } => w.len() / k,
    }
  }

  /// Rows from `r0` times `x` into `out`.
  fn rows(&self, r0: usize, x: &[f32], out: &mut [f32]) {
    match self {
      Self::Q8(q8) => q8.rows(r0, x, out),
      Self::F32 { k, w } => {
        for (o, w) in out.iter_mut().zip(w[r0 * k..].chunks_exact(*k)) {
          *o = dot(w, x);
        }
      }
    }
  }
}

/// Q8_0 weights `[rows][k]` (k a multiple of 32) as int8 values and one f32 scale per 32. A
/// weight is `q · d`, exact in f32, so every product below equals `dot` on the dequantized row.
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

  use super::{OnceLock, Q8};

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
    let v: Vec<f32> = (0..rows * k).map(|i| (i * 7919 % 31) as f32 / 8.0 - 15.0 / 8.0).collect();
    let f32s: Vec<u8> = v.iter().flat_map(|v| v.to_le_bytes()).collect();
    let f16s: Vec<u8> = v.iter().flat_map(|&v| f16::from_f32(v).to_le_bytes()).collect();
    let mut q8 = Vec::new();
    for b in v.as_chunks::<32>().0 {
      q8.extend(f16::from_f32(0.125).to_le_bytes());
      q8.extend(b.map(|v| (v * 8.0) as i8 as u8));
    }

    let x: Vec<f32> = (0..k).map(|i| (i * 31 % 17) as f32 / 8.0 - 1.0).collect();
    let want: Vec<u32> = v.chunks_exact(k).map(|r| super::dot(r, &x).to_bits()).collect();
    for (ty, bytes) in [(Type::F32, f32s), (Type::F16, f16s), (Type::Q8_0, q8)] {
      let g = Gguf::with_tensors(&[("w", &[k, rows], ty, &bytes)]);
      let w = super::Matrix::new(&g.tensor("w", &[k, rows]).unwrap()).unwrap();
      assert_eq!(w.row(69), v[69 * k..], "{ty:?}");
      let [got] = super::matvecs([(&w, &x)]);
      assert_eq!(got.into_iter().map(f32::to_bits).collect::<Vec<_>>(), want, "{ty:?}");
    }
  }

  #[test]
  fn pool_blocks_and_recovers_after_panics() {
    use std::sync::atomic::{AtomicBool, Ordering};
    if super::threads() < 2 {
      return;
    }
    let pool = super::Pool::new();
    let is_worker = || std::thread::current().name().is_some_and(|n| n.starts_with("wavo-cpu-"));

    // Hold the worker until the caller enters the blocking Condvar wait (`done.waiting == true`).
    let blocked = AtomicBool::new(false);
    pool.run_threads(2, &|tid| {
      if tid == 1 {
        assert!(is_worker());
        while !pool.done.waiting.load(Ordering::Acquire) {
          std::thread::yield_now();
        }
        blocked.store(true, Ordering::Relaxed);
      }
    });
    assert!(blocked.load(Ordering::Relaxed));
    assert_eq!(pool.workers.lock().unwrap().len(), 1, "pool should grow lazily");

    // Caller panic on the pool poisons its lock, and later calls still recover it.
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      pool.run_threads(2, &|tid| {
        if tid == 0 {
          panic!("caller boom");
        }
        assert!(is_worker());
      });
    }));
    assert!(res.is_err());

    // Worker panic propagates to the caller and leaves the pool reusable.
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      pool.run_threads(2, &|tid| {
        if tid == 1 {
          assert!(is_worker());
          panic!("worker boom");
        }
      });
    }));
    assert!(res.is_err());

    // Subsequent work still runs on the persistent worker thread (not scoped fallback).
    let reused = AtomicBool::new(false);
    pool.run_threads(2, &|tid| {
      if tid == 1 {
        reused.store(is_worker(), Ordering::Relaxed);
      }
    });
    assert!(reused.load(Ordering::Relaxed), "expected pool worker reuse after panics");
  }
}
