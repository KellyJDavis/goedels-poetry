import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1978, Problem 1

m and n are positive integers with m < n.
The last three decimal digits of 1978ᵐ are the same as the
last three decimal digits of 1978ⁿ.
Find m and n such that m + n has the least possible value.
-/

namespace Imo1978P1

/- determine -/ abbrev solution : ℕ × ℕ := sorry

abbrev ValidPair : ℕ × ℕ → Prop
| (m, n) => 1 ≤ m ∧ m < n ∧ (1978^m) % 1000 = (1978^n) % 1000

theorem imo1978_p1 (m n : ℕ)
    (hmn : (m, n) = solution) :
    ValidPair (m, n) ∧
    (∀ m' n' : ℕ, ValidPair (m', n') → m + n ≤ m' + n') := sorry

end Imo1978P1
