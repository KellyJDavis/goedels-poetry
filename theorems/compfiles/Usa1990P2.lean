import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 1990, Problem 2

A sequence of functions {fₙ(x)} is defined recursively as follows:

                  f₀(x) = 8
                fₙ₊₁(x) = √(x² + 6fₙ(x)).

For each nonnegative integer n, find all real solutions of the equation

                  fₙ(x) = 2x.
-/

namespace Usa1990P2

noncomputable def f : ℕ → ℝ → ℝ
|     0, _ => 8
| n + 1, x => Real.sqrt (x^2 + 6 * f n x)

/- determine -/ abbrev solution_set (n : ℕ) : Set ℝ := sorry

theorem usa1990_p2 (n : ℕ) (x : ℝ) : x ∈ solution_set n ↔ f n x = 2 * x := sorry


end Usa1990P2
