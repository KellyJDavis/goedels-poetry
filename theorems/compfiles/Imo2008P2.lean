import Mathlib.Data.Real.Basic
import Mathlib.Data.Set.Finite.Lattice
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Abel
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# International Mathematical Olympiad 2008, Problem 2
(a) Prove that
          ```
          x^2 / (x-1)^2 + y^2 / (y-1)^2 + z^2 / (z-1)^2 ≥ 1
          ```
for all real numbers `x`,`y`, `z`, each different from 1, and satisfying `xyz = 1`.

(b) Prove that equality holds above for infinitely many triples of rational numbers `x`, `y`, `z`,
each different from 1, and satisfying `xyz = 1`.
-/

namespace Imo2008P2

theorem imo2008_p2a (x y z : ℝ) (h : x * y * z = 1) (hx : x ≠ 1) (hy : y ≠ 1) (hz : z ≠ 1) :
    x ^ 2 / (x - 1) ^ 2 + y ^ 2 / (y - 1) ^ 2 + z ^ 2 / (z - 1) ^ 2 ≥ 1 := sorry

def rationalSolutions :=
  {s : ℚ × ℚ × ℚ | ∃ x y z : ℚ, s = (x, y, z) ∧ x ≠ 1 ∧ y ≠ 1 ∧ z ≠ 1 ∧ x * y * z = 1 ∧
    x ^ 2 / (x - 1) ^ 2 + y ^ 2 / (y - 1) ^ 2 + z ^ 2 / (z - 1) ^ 2 = 1}

theorem imo2008_p2b : Set.Infinite rationalSolutions := sorry



end Imo2008P2
