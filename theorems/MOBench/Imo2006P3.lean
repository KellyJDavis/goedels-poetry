
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2006, Problem 3

Determine the least real number $M$ such that
$$
\left| ab(a^2 - b^2) + bc(b^2 - c^2) + ca(c^2 - a^2) \right|
≤ M (a^2 + b^2 + c^2)^2
$$
for all real numbers $a$, $b$, $c$.
-/
noncomputable abbrev solution : ℝ := 9 * Real.sqrt 2 / 32

theorem imo2006_p3 :
    IsLeast
      { M | (∀ a b c : ℝ,
              |a * b * (a ^ 2 - b ^ 2) + b * c * (b ^ 2 - c ^ 2) + c * a * (c ^ 2 - a ^ 2)| ≤
              M * (a ^ 2 + b ^ 2 + c ^ 2) ^ 2) } solution := by sorry
