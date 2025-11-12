
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2010 A3

Fix a positive integer $N$, a totally ordered commutative ring $R$, and an element $c \ge 0$.
Consider all $2N$-periodic sequences $(x_n)_{n \ge 0}$ such that for any $n$,
$$ x_n + x_{n + 1} + x_{n + 2} \le 2c. $$
Determine the maximum possible value of
$$ \sum_{k = 0}^{2N-1} (x_k x_{k + 2} + x_{k + 1} x_{k + 3}). $$
-/
/- special open -/ open Finset
variable (R : Type*) [LinearOrderedCommRing R]

/--
A sequence `x` is a "good periodic sequence" if it satisfies the conditions of the problem:
- `nonneg`: All its elements are non-negative.
- `good_sum`: The sum of any three consecutive elements is at most `2c`.
- `periodic`: The sequence is periodic with period `2N`.
-/
structure IsGoodPeriodicSeq (c : R) (N : ℕ) where
  x : ℕ → R
  nonneg : ∀ i, 0 ≤ x i
  good_sum : ∀ i, x i + x (i + 1) + x (i + 2) ≤ 2 • c
  periodic : ∀ k, x (k + 2 * N) = x k

/-- The expression to be maximized. -/
def targetSum (x : ℕ → R) (N : ℕ) : R :=
  ∑ i ∈ range (2 * N), (x i * x (i + 2) + x (i + 1) * x (i + 3))

/--
The maximum value of the target sum is $2Nc^2$.
`IsGreatest S m` means `m` is the maximum value of the set `S`.
-/
theorem imo_sl_2010_A3 {c : R} (hc : 0 ≤ c) {N : ℕ} (hN : 0 < N) :
  IsGreatest (Set.range fun (s : IsGoodPeriodicSeq R c N) ↦ targetSum R s.x N) (2 * N • c ^ 2) :=
  by sorry
