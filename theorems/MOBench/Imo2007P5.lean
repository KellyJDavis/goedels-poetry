
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2007, Problem 5

Let a and b be positive integers. Show that if 4ab - 1 divides (4a² - 1)²
then a = b.
-/
theorem imo2007_p5 (a b : ℤ) (ha : 0 < a) (hb : 0 < b)
    (hab : 4 * a * b - 1 ∣ (4 * a^2 - 1)^2) : a = b := by sorry
