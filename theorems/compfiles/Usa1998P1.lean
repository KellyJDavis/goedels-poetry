import Mathlib.Algebra.BigOperators.Group.Finset
import Mathlib.Algebra.BigOperators.Intervals
import Mathlib.Data.Int.ModEq
import Mathlib.Data.Finset.Interval
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 1998, Problem 1

Suppose that the set { 1, 2, ..., 1998 } has been partitioned into disjoint
pairs {aᵢ, bᵢ}, where 1 ≤ i ≤ 999, so that for all i, |aᵢ - bᵢ| = 1 or 6.

Prove that the sum

  |a₁ - b₁| + |a₂ - b₂| + ... + |a₉₉₉ - b₉₉₉|

ends in the digit 9.
-/

namespace Usa1998P1

/--
 `ab 0 i` is aᵢ and `ab 1 i` is `bᵢ`
-/
theorem usa1998_p1
    (ab : Fin 2 → Fin 999 → Finset.Icc 1 1998)
    (hab : (ab.uncurry).Bijective)
    (habd : ∀ i : Fin 999,
              |(ab 0 i : ℤ) - (ab 1 i : ℤ)| = 1 ∨
              |(ab 0 i : ℤ) - (ab 1 i : ℤ)| = 6) :
    (∑ i : Fin 999, |(ab 0 i : ℤ) - (ab 1 i : ℤ)|) % 10 = 9 := sorry

end Usa1998P1
