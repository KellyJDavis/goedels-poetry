import Mathlib.Tactic
import Mathlib.Data.Nat.Choose.Basic

/-!
# International Mathematical Olympiad 1972, Problem 3

Let m and n be non-negative integers. Prove that

     (2m)!(2n)! / (m!n!(m + n)!)

is an integer.
-/

namespace Imo1972P3

open scoped Nat

theorem imo1972_p3 (m n : ℕ) :
    m ! * n ! * (m + n)! ∣ (2 * m)! * (2 * n)! := sorry


end Imo1972P3
