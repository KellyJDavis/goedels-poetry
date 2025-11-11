import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1962, Problem 2

Determine all real numbers x which satisfy

  √(3 - x) - √(x + 1) > 1/2.
-/

namespace Imo1962P2

/- determine -/ abbrev SolutionSet : Set ℝ := sorry

theorem imo1962_p2 (x : ℝ) :
    x ∈ SolutionSet ↔
    x ≤ 3 ∧ -1 ≤ x ∧ 1/2 < √(3 - x) - √(x + 1) := sorry


end Imo1962P2
