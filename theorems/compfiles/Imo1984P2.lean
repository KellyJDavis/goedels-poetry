import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1984, Problem 2

Find a pair of positive integers a and b such that

 (i) ab(a + b) is not divisible by 7.
 (ii) (a + b)⁷ - a⁷ - b⁷ is divisible by 7⁷.
-/

namespace Imo1984P2

/- determine -/ abbrev a : ℤ := sorry
/- determine -/ abbrev b : ℤ := sorry

theorem imo1984_p2 :
    (0 < a) ∧ (0 < b) ∧
    (¬ 7 ∣ a * b * (a + b)) ∧
    7^7 ∣ (a + b)^7 - a^7 - b^7 := sorry


end Imo1984P2
