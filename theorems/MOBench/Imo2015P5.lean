
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2015, Problem 5

Determine all functions f : ℝ → ℝ that satisfy

  f(x + f(x + y)) + f(xy) = x + f(x + y) + yf(x)

for all x,y.
-/
abbrev SolutionSet : Set (ℝ → ℝ) := { fun x ↦ x, fun x ↦ 2 - x }

theorem imo2015_p5 (f : ℝ → ℝ) :
    f ∈ SolutionSet ↔
    ∀ x y, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x := by sorry
