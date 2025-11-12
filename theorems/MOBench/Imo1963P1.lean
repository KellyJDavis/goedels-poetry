
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1963, Problem 1

Find all real roots of the equation

  √(x²-p) + 2√(x²-1) = x

where *p* is a real parameter.
-/
abbrev f (p : ℝ) : Set ℝ :=
  if p ≥ 0 ∧ p ≤ (4 : ℝ) / 3
  then { (4 - p) / (2 * Real.sqrt (4 - 2 * p)) }
  else ∅

theorem imo1963_p1 : ∀ (p x : ℝ), (x ^ 2 - p) ≥ 0 → (x ^ 2 - 1) ≥ 0 →
  (Real.sqrt (x ^ 2 - p) + 2 * Real.sqrt (x ^ 2 - 1) = x ↔ (x ∈ f p)) := by sorry
