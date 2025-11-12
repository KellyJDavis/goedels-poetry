
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
Indian Mathematical Olympiad 1998, problem 1

(a) Show that the product of two numbers of the form a² + 3b² is again of that form.
-/
theorem india1998_p1a (a₁ a₂ b₁ b₂ : ℤ) :
    (∃ a₃ b₃, (a₁^2 + 3 * b₁^2) * (a₂^2 + 3 * b₂^2) = (a₃^2 + 3 * b₃^2)) := by sorry
