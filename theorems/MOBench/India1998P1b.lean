
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
Indian Mathematical Olympiad 1998, problem 1

(b) If an integer n is such that 7n is of the form a² + 3b², prove that n is also of that form.
-/
theorem india1998_p1b (n a b : ℤ) (hn : a^2 + 3 * b^2 = 7 * n) :
    (∃ a b : ℤ, a^2 + 3 * b^2 = n) := by sorry
