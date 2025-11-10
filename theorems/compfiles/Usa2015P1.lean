import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 2015, Problem 1

Solve in integers the equation x² + xy + y² = ((x + y) / 3 + 1)³.
-/

namespace Usa2015P1

/- determine -/ abbrev SolutionSet : Set (ℤ × ℤ) := sorry

theorem usa2015_p1 (x y : ℤ) :
    ⟨x, y⟩ ∈ SolutionSet ↔
    x^2 + x * y + y^2 = ((x + y) / (3 : ℚ) + 1)^3 := sorry


end Usa2015P1
