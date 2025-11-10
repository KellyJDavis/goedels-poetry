import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Data.Rat.Defs
import Mathlib.Order.WellFounded
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.WLOG

/-!
# International Mathematical Olympiad 1988, Problem 6

If a and b are two natural numbers such that a*b+1 divides a^2 + b^2,
show that their quotient is a perfect square.
-/

namespace Imo1988P6

theorem imo1988_p6 {a b : ℕ} (h : a * b + 1 ∣ a ^ 2 + b ^ 2) :
    ∃ d, d ^ 2 = (a ^ 2 + b ^ 2) / (a * b + 1) := sorry



end Imo1988P6
