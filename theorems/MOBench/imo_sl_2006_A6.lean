
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
International Mathematical Olympiad 2006, Problem 3

Find the smallest M ∈ ℝ such that for any a, b, c ∈ ℝ,
|ab(a² - b²) + bc(b² - c²) + ca(c² - a²)| ≤ M(a² + b² + c²)².
-/
def good [LinearOrderedCommRing R] (M : R) :=
  ∀ a b c : R, |a * b * (a ^ 2 - b ^ 2) + b * c * (b ^ 2 - c ^ 2) + c * a * (c ^ 2 - a ^ 2)|
    ≤ M * (a ^ 2 + b ^ 2 + c ^ 2) ^ 2

theorem good_iff : good M ↔ 9 * √2 ≤ 32 * M := by sorry
