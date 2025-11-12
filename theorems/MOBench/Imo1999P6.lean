
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1999, Problem 6

Determine all functions f : ℝ → ℝ such that

  f(x - f(y)) = f(f(y)) + xf(y) + f(x) - 1

for all x,y ∈ ℝ.
-/
abbrev SolutionSet : Set (ℝ → ℝ) := { fun x ↦ 1 - x^2 / 2 }

theorem imo1999_p6 (f : ℝ → ℝ) :
    f ∈ SolutionSet ↔
    ∀ x y, f (x - f y) = f (f y) + x * f y + f x - 1 := by sorry
