
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2017, Problem 2

Find all functions `f : ℝ → ℝ` that satisfy

  ∀ x,y ∈ ℝ, f(f(x)f(y)) + f(x + y) = f(xy).
-/
abbrev solution_set : Set (ℝ → ℝ) :=
  { fun _ ↦ 0, fun x ↦ x - 1, fun x ↦ 1 - x }

theorem imo2017_p2 (f : ℝ → ℝ) :
    f ∈ solution_set ↔ ∀ x y, f (f x * f y) + f (x + y) = f (x * y) := by sorry
