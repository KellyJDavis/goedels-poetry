import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2017, Problem 2

Find all functions `f : ℝ → ℝ` that satisfy

  ∀ x,y ∈ ℝ, f(f(x)f(y)) + f(x + y) = f(xy).
-/

namespace Imo2017P2

/- determine -/ abbrev solution_set : Set (ℝ → ℝ) := sorry

theorem imo2017_p2 (f : ℝ → ℝ) :
    f ∈ solution_set ↔ ∀ x y, f (f x * f y) + f (x + y) = f (x * y) := sorry


end Imo2017P2
