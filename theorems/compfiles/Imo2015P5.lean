import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2015, Problem 5

Determine all functions f : ℝ → ℝ that satisfy

  f(x + f(x + y)) + f(xy) = x + f(x + y) + yf(x)

for all x,y.
-/

namespace Imo2015P5

/- determine -/ abbrev SolutionSet : Set (ℝ → ℝ) := sorry

theorem imo2015_p5 (f : ℝ → ℝ) :
    f ∈ SolutionSet ↔
    ∀ x y, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x := sorry


end Imo2015P5
