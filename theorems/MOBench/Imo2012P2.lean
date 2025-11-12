
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2012, Problem 2

Let a₂, a₃, ..., aₙ be positive reals with product 1, where n ≥ 3.
Show that
  (1 + a₂)²(1 + a₃)³...(1 + aₙ)ⁿ > nⁿ.
-/
theorem imo2012_p2 (n : ℕ) (hn : 3 ≤ n) (a : Finset.Icc 2 n → ℝ)
    (apos : ∀ i, 0 < a i) (aprod : ∏ i, a i = 1) :
    (n:ℝ)^n < ∏ i, (1 + a i)^i.val := by sorry
