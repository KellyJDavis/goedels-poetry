import Mathlib.Data.Nat.Digits
import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 2003, Problem 1

Prove that for every positive integer n there exists an n-digit
number divisible by 5ⁿ, all of whose digits are odd.
-/

namespace Usa2003P1

theorem usa2003_p1 (n : ℕ) :
    ∃ m, (Nat.digits 10 m).length = n ∧
          5^n ∣ m ∧ (Nat.digits 10 m).all (Odd ·) := sorry

end Usa2003P1
