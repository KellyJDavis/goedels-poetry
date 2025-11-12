
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1959, Problem 2

For what real values of x is

 √(x+√(2x-1)) + √(x-√(2x-1)) = A,

given:
  (a) A = √2
  (b) A = 1
  (c) A = 2,
where only non-negative real numbers are admitted for square roots?
-/
/- special open -/ open Set






def IsGood (x A : ℝ) : Prop :=
  sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = A ∧ 0 ≤ 2 * x - 1 ∧
    0 ≤ x + sqrt (2 * x - 1) ∧ 0 ≤ x - sqrt (2 * x - 1)

variable {x A : ℝ}

abbrev solution_set_one : Set ℝ := ∅

theorem imo1959_p2b : IsGood x 1 ↔ x ∈ solution_set_one := by sorry
