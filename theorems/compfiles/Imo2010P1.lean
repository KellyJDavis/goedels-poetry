import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2010, Problem 1

Determine all functions f : ℝ → ℝ such that for all x,y ∈ ℝ,

               f(⌊x⌋y) = f(x)⌊f(y)⌋.
-/

namespace Imo2010P1

/- determine -/ abbrev solution_set : Set (ℝ → ℝ) := sorry

theorem imo2010_p1 (f : ℝ → ℝ) :
    f ∈ solution_set ↔ ∀ x y, f (⌊x⌋ * y) = f x * ⌊f y⌋ := sorry


end Imo2010P1
