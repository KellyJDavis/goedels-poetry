import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 1993, Problem 1

For each integer n ≥ 2, determine whether a or b is larger,
where a and b are positive real numbers satisfying

            aⁿ = a + 1,     b²ⁿ = b + 3a.
-/

namespace Usa1993P1

/- determine -/ abbrev a_is_larger : ℕ → Bool := sorry

theorem usa1993_p1 (n : ℕ) (hn : 2 ≤ n) (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
    (han : a^n = a + 1) (hbn : b^(2 * n) = b + 3 * a) :
    if a_is_larger n then b < a else a < b := sorry


end Usa1993P1
