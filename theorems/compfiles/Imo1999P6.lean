import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1999, Problem 6

Determine all functions f : ℝ → ℝ such that

  f(x - f(y)) = f(f(y)) + xf(y) + f(x) - 1

for all x,y ∈ ℝ.
-/

namespace Imo1999P6

/- determine -/ abbrev SolutionSet : Set (ℝ → ℝ) := sorry

theorem imo1999_p6 (f : ℝ → ℝ) :
    f ∈ SolutionSet ↔
    ∀ x y, f (x - f y) = f (f y) + x * f y + f x - 1 := sorry


end Imo1999P6
