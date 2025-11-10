import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1992, Problem 2

Determine all functions f : ℝ → ℝ such that
for all x,y ∈ ℝ, f(x² + f(y)) = y + (f(x))².
-/

namespace Imo1992P2

/- determine -/ abbrev solution_set : Set (ℝ → ℝ) := sorry

theorem imo1992_p2 (f : ℝ → ℝ) :
    f ∈ solution_set ↔
    ∀ x y, f (x^2 + f y) = y + f x ^ 2 := sorry


end Imo1992P2
