import Mathlib.Data.Int.ModEq
import Mathlib.Data.ZMod.Basic
import Mathlib.Tactic

/-!
Indian Mathematical Olympiad 1998, problem 1

(a) Show that the product of two numbers of the form a² + 3b² is again of that form.
(b) If an integer n is such that 7n is of the form a² + 3b², prove that n is also of that form.
-/

namespace India1998P1

theorem india1998_p1a (a₁ a₂ b₁ b₂ : ℤ) :
    (∃ a₃ b₃, (a₁^2 + 3 * b₁^2) * (a₂^2 + 3 * b₂^2) = (a₃^2 + 3 * b₃^2)) := sorry

theorem india1998_p1b (n a b : ℤ) (hn : a^2 + 3 * b^2 = 7 * n) :
    (∃ a b : ℤ, a^2 + 3 * b^2 = n) := sorry


end India1998P1
