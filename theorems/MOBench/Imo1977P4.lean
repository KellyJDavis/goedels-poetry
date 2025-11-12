
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1977, Problem 4

Define f(x) = 1 - a cos x - b sin x - A cos 2x - B sin 2x,
where a, b, A, B are real constants.
Suppose that f(x) ≥ 0 for all real x.
Prove that a^2 + b^2 ≤ 2 and A^2 + B^2 ≤ 1.
-/
theorem imo1977_p4 (f : ℝ → ℝ) (a b A B : ℝ)
  (h₀ : ∀ x, f x =
             1 - a * Real.cos x - b * Real.sin x - A * Real.cos (2 * x) - B * Real.sin (2 * x))
  (h₁ : ∀ x, f x ≥ 0) :
    a ^ 2 + b ^ 2 ≤ 2 ∧ A ^ 2 + B ^ 2 ≤ 1 := by sorry
