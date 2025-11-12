
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 2022, Problem 3

Let ℝ+ be the set of all positive real numbers. Find all
functions ℝ+ → ℝ+ such that for all x, y ∈ ℝ+ we have

   f(x) = f(f(f(x)) + y) + f(xf(y))f(x+y).
-/
abbrev PosReal : Type := { x : ℝ // 0 < x }
notation "ℝ+" => PosReal

abbrev solution_set : Set (ℝ+ → ℝ+) :=
  { f : ℝ+ → ℝ+ | ∃ c : ℝ+, f = fun x ↦ c / x }

theorem usa2022_p3 (f : ℝ+ → ℝ+) :
  f ∈ solution_set ↔
    (∀ x y : ℝ+, f x = f (f (f x) + y) + f (x * f y) * f (x + y)) := by sorry
