"""Continuous batching for open-dreamer rollouts, and what it is actually worth.

The released `KVCache.index` is a scalar shared by the whole batch
(`models.py:43`). Every row must therefore sit at the same rollout step, so a
served batch can only be built from sessions that start together and never
diverge. `ragged_kv` in bench/patches.py makes the index `(B,)`, which is what
lets a finished session be evicted from one slot and a new one admitted into it
without touching the other rows.

The throughput ceiling is already known and is not large. From the measured
batch sweep (bench/RESULTS.md, log bzce85sia) the frame cost is affine to
within 2%:

    t(B) = 13.33 ms + 12.93 ms x B

so aggregate throughput saturates at 1000/12.93 = 77.4 fps, exactly 1.99x over
B=1, and B=8 already reaches 89% of that. Continuous batching cannot beat that
ceiling. What it changes is how much of the ceiling a *real* arrival pattern
gets to keep.

The thing that makes this matter is padding. A JAX rollout is compiled for a
fixed batch width B and idle slots are padded, not skipped, so a step costs
t(B) whether 8 slots are busy or 1. Throughput is therefore

    fps = occupancy x B / t(B)

and occupancy is entirely a scheduling property. Static batching drains a whole
cohort before admitting anyone, so its occupancy decays across the cohort's
life; continuous batching refills a slot the step after it frees.

This module has two parts:
  * `ContinuousBatchScheduler` -- slot admission/eviction over a ragged cache.
  * `simulate()` -- a discrete-event comparison of static vs continuous under
    Poisson arrivals, driven by the *measured* cost model above rather than by
    an assumed one.
"""
from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass, field

# Measured on H100, recommended stack, log bzce85sia. Fit error <=2% at every
# B in {1,2,4,8,16}; <=0.1% for B>=4.
FIXED_MS = 13.33
MARGINAL_MS = 12.93

# Ragged indices are not free. Same-container A/B at B=8 (log b2fyjwe06):
# scalar index 116.733 ms/frame, ragged 122.915 ms -- 5.3% slower. The whole
# cost is in `ladder` (107.877 -> 114.490 ms), i.e. the vmap'd per-row
# dynamic_update_slice; `dyn_fwd` is marginally *faster* (24.299 -> 23.650).
# Continuous batching must pay this, so the simulation charges it rather than
# comparing an idealised continuous policy against a real static one.
RAGGED_PENALTY = 122.915 / 116.733


def step_ms(width: int, ragged: bool = False) -> float:
    """Cost of one frame step for a rollout compiled at batch width `width`.

    Note this takes the *compiled width*, not the number of active sessions.
    Padded slots are computed, not skipped, which is the entire reason
    occupancy determines throughput.
    """
    t = FIXED_MS + MARGINAL_MS * width
    return t * RAGGED_PENALTY if ragged else t


# ---------------------------------------------------------------------------
# the scheduler itself
# ---------------------------------------------------------------------------

@dataclass
class Session:
    sid: int
    frames_total: int
    frames_done: int = 0
    arrived_ms: float = 0.0
    admitted_ms: float | None = None
    finished_ms: float | None = None

    @property
    def done(self) -> bool:
        return self.frames_done >= self.frames_total


class ContinuousBatchScheduler:
    """Fixed-width batch of slots, each holding an independent session.

    Correctness rests on one property of `ragged_kv`, which
    `check_ragged_kv_equivalence` verifies on GPU at 0.00e+00: a batch whose
    rows hold different indices produces the same result as running each row
    separately. Admission is then just `index[slot] = 0` -- no need to clear
    the slot's K/V, because the `written = age <= index - 1` term of the mask
    already excludes every stale entry.
    """

    def __init__(self, width: int):
        self.width = width
        self.slots: list[Session | None] = [None] * width
        self.waiting: list[Session] = []

    @property
    def active(self) -> int:
        return sum(s is not None for s in self.slots)

    @property
    def occupancy(self) -> float:
        return self.active / self.width

    def submit(self, s: Session) -> None:
        self.waiting.append(s)

    def admit(self, now_ms: float) -> list[int]:
        """Fill every free slot from the queue. Returns the slots filled.

        The caller must reset `cache.index[slot] = 0` for each returned slot.
        """
        filled = []
        for i, occupant in enumerate(self.slots):
            if occupant is None and self.waiting:
                s = self.waiting.pop(0)
                s.admitted_ms = now_ms
                self.slots[i] = s
                filled.append(i)
        return filled

    def retire(self, now_ms: float) -> list[int]:
        """Free every slot whose session has produced all its frames."""
        freed = []
        for i, s in enumerate(self.slots):
            if s is not None and s.done:
                s.finished_ms = now_ms
                self.slots[i] = None
                freed.append(i)
        return freed

    def advance(self) -> None:
        """One frame produced for every occupied slot."""
        for s in self.slots:
            if s is not None:
                s.frames_done += 1


def reset_slots(cache_index, slots):
    """`index[slot] = 0` for each admitted slot, on a ragged `(B,)` index.

    Split out because it is the one line that a scalar index cannot express:
    with `index` a scalar there is no way to rewind one row without rewinding
    every row, which is precisely why the released cache forces static batching.
    """
    import jax.numpy as jnp
    idx = cache_index
    for s in slots:
        idx = idx.at[s].set(0)
    return jnp.asarray(idx)


# ---------------------------------------------------------------------------
# static batching, the thing we are comparing against
# ---------------------------------------------------------------------------

class StaticBatchScheduler(ContinuousBatchScheduler):
    """A cohort is admitted together and no one else joins until all are done.

    This is what a scalar `index` forces. It is not a strawman: with one shared
    index there is no correct way to admit a session mid-flight, since the new
    row would inherit the cohort's position.
    """

    def __init__(self, width: int):
        super().__init__(width)
        self._draining = False

    def admit(self, now_ms: float) -> list[int]:
        if self._draining:
            return []
        # Only start a cohort once the batch is fully free.
        if self.active:
            return []
        filled = super().admit(now_ms)
        if filled:
            self._draining = True
        return filled

    def retire(self, now_ms: float) -> list[int]:
        freed = super().retire(now_ms)
        if self._draining and self.active == 0:
            self._draining = False
        return freed


# ---------------------------------------------------------------------------
# discrete-event comparison
# ---------------------------------------------------------------------------

@dataclass
class SimResult:
    name: str
    width: int
    wall_ms: float
    frames: int
    completed: int
    occ_samples: list[float] = field(default_factory=list)
    waits_ms: list[float] = field(default_factory=list)

    @property
    def fps(self) -> float:
        return self.frames / (self.wall_ms / 1000.0)

    @property
    def occupancy(self) -> float:
        return statistics.fmean(self.occ_samples) if self.occ_samples else 0.0

    @property
    def p50_wait(self) -> float:
        return statistics.median(self.waits_ms) if self.waits_ms else 0.0

    @property
    def p95_wait(self) -> float:
        if not self.waits_ms:
            return 0.0
        return sorted(self.waits_ms)[int(0.95 * (len(self.waits_ms) - 1))]


def simulate(sched_cls, width: int, arrivals, horizon_ms: float) -> SimResult:
    """Step the batch until `horizon_ms`, admitting from `arrivals` as they land.

    `arrivals` is a sorted list of (time_ms, Session). One step costs
    `step_ms(width)` regardless of occupancy -- the padding fact that makes
    this comparison meaningful at all.
    """
    sched = sched_cls(width)
    pending = list(arrivals)
    now = 0.0
    frames = 0
    completed = 0
    res = SimResult(sched_cls.__name__, width, 0.0, 0, 0)

    while now < horizon_ms:
        while pending and pending[0][0] <= now:
            sched.submit(pending.pop(0)[1])
        sched.admit(now)
        if sched.active == 0:
            # Nothing to run. Jump to the next arrival rather than burning a
            # step on an entirely empty batch -- charging for idle time the
            # scheduler could not have used would flatter neither policy.
            if not pending:
                break
            now = max(now, pending[0][0])
            continue

        res.occ_samples.append(sched.occupancy)
        sched.advance()
        frames += sched.active
        # Continuous batching needs the ragged index, so it pays the measured
        # 5.3%. Static batching runs on the released scalar-index path.
        now += step_ms(width, ragged=sched_cls is ContinuousBatchScheduler)
        for s in sched.slots:
            if s is not None and s.done and s.admitted_ms is not None:
                res.waits_ms.append(s.admitted_ms - s.arrived_ms)
        completed += len(sched.retire(now))

    res.wall_ms = now
    res.frames = frames
    res.completed = completed
    return res


def poisson_arrivals(rate_per_s: float, n: int, frames: int, seed: int,
                     length_cv: float = 0.0):
    """Exponential inter-arrivals. Uses `random` seeded explicitly so the two
    policies see byte-identical arrival streams.

    `length_cv` is the coefficient of variation of session length. It is the
    parameter that decides this whole comparison, and setting it to 0 quietly
    hands the win to static batching: with every session exactly `frames` long
    a cohort drains all at once and no slot ever sits idle waiting for a
    straggler. Real sessions are not uniform -- a user closes the tab, a policy
    hits a terminal state -- and every short session in a static cohort holds
    its slot idle until the longest one in that cohort finishes.
    """
    import random
    rng = random.Random(seed)
    out, t = [], 0.0
    for sid in range(n):
        t += rng.expovariate(rate_per_s) * 1000.0
        if length_cv > 0:
            # Lognormal: positive, right-skewed, and the long tail is exactly
            # the straggler that pins a static cohort's slots.
            sigma = (math.log(1 + length_cv ** 2)) ** 0.5
            mu = math.log(frames) - sigma ** 2 / 2
            n_frames = max(1, int(rng.lognormvariate(mu, sigma)))
        else:
            n_frames = frames
        out.append((t, Session(sid=sid, frames_total=n_frames, arrived_ms=t)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--rate", type=float, default=0.55, help="sessions/sec")
    ap.add_argument("--frames", type=int, default=120, help="frames per session")
    ap.add_argument("--sessions", type=int, default=400)
    ap.add_argument("--horizon-s", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--length-cv", type=float, default=0.6,
                    help="coefficient of variation of session length")
    args = ap.parse_args()

    print(f"cost model  t(B) = {FIXED_MS} ms + {MARGINAL_MS} ms x B   "
          f"(measured, log bzce85sia)")
    print(f"arrivals    Poisson {args.rate}/s, {args.frames} frames/session, "
          f"seed {args.seed}, length CV {args.length_cv}")
    cap = (1000 / MARGINAL_MS) / args.frames
    print(f"capacity    ~{cap:.2f} sessions/s at {args.frames} frames each"
          f"  -> offered load {args.rate / cap * 100:.0f}%")
    print(f"ceiling     {1000 / MARGINAL_MS:.1f} fps aggregate\n")

    print(f"  {'width':>6}{'policy':>12}{'occupancy':>11}{'fps':>9}"
          f"{'of ceiling':>12}{'p50 wait':>11}{'p95 wait':>11}")
    for w in args.width:
        rows = []
        for cls in (StaticBatchScheduler, ContinuousBatchScheduler):
            arr = poisson_arrivals(args.rate, args.sessions, args.frames,
                                   args.seed, args.length_cv)
            r = simulate(cls, w, arr, args.horizon_s * 1000.0)
            rows.append(r)
            label = "static" if cls is StaticBatchScheduler else "continuous"
            print(f"  {w:>6}{label:>12}{r.occupancy * 100:10.1f}%{r.fps:9.1f}"
                  f"{r.fps / (1000 / MARGINAL_MS) * 100:11.0f}%"
                  f"{r.p50_wait:10.0f}ms{r.p95_wait:10.0f}ms")
        if rows[0].fps > 0:
            print(f"  {'':>6}{'-> gain':>12}{'':11}{rows[1].fps / rows[0].fps:8.2f}x")
    print("\n  Both policies pay the same t(B) per step. The whole difference is")
    print("  occupancy, which is why the gain is a scheduling result and not a")
    print("  kernel one -- and why it cannot exceed the 1.99x batching ceiling.")


if __name__ == "__main__":
    main()
