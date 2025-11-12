
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1961, Problem 1.

Given constants a and b, solve the system of equations

             x + y + z = a
          x² + y² + z² = b²
                    xy = z²

for x,y,z. Give the conditions that a and b must satisfy so that
the solutions x,y,z are distinct positive numbers.
-/
abbrev IsSolution (a b x y z : ℝ) : Prop :=
    x + y + z = a ∧
    x^2 + y^2 + z^2 = b^2 ∧
    x * y = z^2

abbrev xyz_of_ab (a b : ℝ) : Set (ℝ × ℝ × ℝ) :=
  { p | let ⟨x,y,z⟩ := p
        z = (a^2 - b^2) / (2 * a) ∧
        ∀ m, (m - x) * (m - y) =
              m^2 - (a^2 + b^2) / (2 * a) * m + ((a^2 - b^2) / (2 * a))^2 }

theorem imo1961_p1a (a b x y z : ℝ) :
    ⟨x,y,z⟩ ∈ xyz_of_ab a b ↔ IsSolution a b x y z := by sorry
