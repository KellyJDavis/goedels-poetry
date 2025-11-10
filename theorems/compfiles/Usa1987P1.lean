import Mathlib.Algebra.QuadraticDiscriminant
import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 1987, Problem 1

Determine all solutions to

     (m² + n)(m + n²) = (m - n)³

where m and n are non-zero integers.
-/

namespace Usa1987P1

/- determine -/ abbrev solution_set : Set (ℤ × ℤ) := sorry

theorem usa1987_p1 (m n : ℤ) :
    (m, n) ∈ solution_set ↔
    m ≠ 0 ∧ n ≠ 0 ∧ (m^2 + n) * (m + n^2) = (m - n)^3 := sorry


end Usa1987P1
