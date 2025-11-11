import Mathlib.Data.Real.Sqrt

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

open Set Real

namespace Imo1959P2

def IsGood (x A : ℝ) : Prop :=
  sqrt (x + sqrt (2 * x - 1)) + sqrt (x - sqrt (2 * x - 1)) = A ∧ 0 ≤ 2 * x - 1 ∧
    0 ≤ x + sqrt (2 * x - 1) ∧ 0 ≤ x - sqrt (2 * x - 1)

variable {x A : ℝ}

/- determine -/ abbrev solution_set_sqrt2 : Set ℝ := sorry

theorem imo1959_p2a : IsGood x (sqrt 2) ↔ x ∈ solution_set_sqrt2 := sorry

/- determine -/ abbrev solution_set_one : Set ℝ := sorry

theorem imo1959_p2b : IsGood x 1 ↔ x ∈ solution_set_one := sorry

/- determine -/ abbrev solution_set_two : Set ℝ := sorry

theorem imo1959_p2c : IsGood x 2 ↔ x ∈ solution_set_two := sorry


end Imo1959P2
