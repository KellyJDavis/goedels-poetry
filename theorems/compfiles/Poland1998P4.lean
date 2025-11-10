import Mathlib.Data.Nat.ModEq
import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

/-!
Polish Mathematical Olympiad 1998, Problem 4

Prove that the sequence {a_n} defined by a_1 = 1 and

     a_n = a_{n - 1} + a_{⌊n/2⌋}        n = 2,3,4,...

contains infinitely many integers divisible by 7.

-/

namespace Poland1998P4

def a : ℕ → ℕ
| 0 => 1 -- unused dummy value
| 1 => 1
| Nat.succ n =>
    have _ : (n.succ / 2) < n.succ := Nat.div_lt_self' n 0
    a n + a (n.succ / 2)

theorem poland1998_p4 : Set.Infinite { n | 7 ∣ a n } := sorry


end Poland1998P4
