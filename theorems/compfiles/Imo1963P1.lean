import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# International Mathematical Olympiad 1963, Problem 1

Find all real roots of the equation

  √(x²-p) + 2√(x²-1) = x

where *p* is a real parameter.
-/

namespace Imo1963P1

/- determine -/ abbrev f (p : ℝ) : Set ℝ := sorry

theorem imo1963_p1 : ∀ (p x : ℝ), (x ^ 2 - p) ≥ 0 → (x ^ 2 - 1) ≥ 0 →
  (Real.sqrt (x ^ 2 - p) + 2 * Real.sqrt (x ^ 2 - 1) = x ↔ (x ∈ f p)) := sorry


end Imo1963P1
