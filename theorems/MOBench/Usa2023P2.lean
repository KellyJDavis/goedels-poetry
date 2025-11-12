
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 2023, Problem 2

Let ℝ+ be the set of positive real numbers.
Find all functions f: ℝ+ → ℝ+ that satisfy the equation

  f(x⬝y + f(x)) = x⬝f(y) + 2

for all x,y ∈ ℝ+.
-/
abbrev PosReal : Type := { x : ℝ // 0 < x }

notation "ℝ+" => PosReal

abbrev solution_set : Set (ℝ+ → ℝ+) := { fun x ↦ x + 1 }

theorem usa2023_p2 (f : ℝ+ → ℝ+) :
    f ∈ solution_set ↔
    ∀ x y, f (x * y + (f x)) = x * (f y) + ⟨2, two_pos⟩ := by sorry
