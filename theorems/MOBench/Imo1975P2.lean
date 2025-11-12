
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1975, Problem 2

Let a1 < a2 < a3 < ... be positive integers.
Prove that for every i >= 1,
there are infinitely many a_n that can be written in the form a_n = ra_i + sa_j,
with r, s positive integers and j > i.
-/
theorem imo1975_p2 (a : ℕ → ℤ)
  (apos : ∀ i : ℕ, 0 < a i)
  (ha : ∀ i : ℕ, a i < a (i + 1)) :
    ( ∀ i n0 : ℕ ,
      ∃ n, n0 ≤ n ∧
      ∃ r s : ℕ,
      ∃ j : ℕ,
        a n = r * a i + s * a j ∧
        i < j ∧
        0 < r ∧
        0 < s ):= by sorry
