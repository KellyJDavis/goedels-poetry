import Mathlib.Analysis.SpecialFunctions.Trigonometric.Complex
import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1961, Problem 3

Solve the equation

  cosⁿ x - sinⁿ x = 1,

where n is a given positive integer.
-/

namespace Imo1961P3

open Real

/- determine -/ abbrev solutionSet (n : ℕ+) : Set ℝ := sorry

theorem imo1961_p3 {n : ℕ} {x : ℝ} (npos : 0 < n) :
    x ∈ solutionSet ⟨n, npos⟩ ↔
    (cos x) ^ n - (sin x) ^ n = 1 := sorry

end Imo1961P3
