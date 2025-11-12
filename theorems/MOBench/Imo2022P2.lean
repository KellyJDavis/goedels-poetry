
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2022, Problem 2

Let ℝ+ be the set of positive real numbers.
Determine all functions f: ℝ+ → ℝ+ such that
for each x ∈ ℝ+, there is exactly one y ∈ ℝ+
satisfying

  x·f(y) + y·f(x) ≤ 2
-/
abbrev PosReal : Type := { x : ℝ // 0 < x }

notation "ℝ+" => PosReal

abbrev solution_set : Set (ℝ+ → ℝ+) := { fun x ↦ 1 / x }

theorem imo2022_p2 (f : ℝ+ → ℝ+) :
    f ∈ solution_set ↔
    ∀ x, ∃! y, x * f y + y * f x ≤ ⟨2, two_pos⟩ := by sorry
