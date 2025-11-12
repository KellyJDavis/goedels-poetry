
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1972, Problem 3

Let m and n be non-negative integers. Prove that

     (2m)!(2n)! / (m!n!(m + n)!)

is an integer.
-/
theorem imo1972_p3 (m n : ℕ) :
    m ! * n ! * (m + n)! ∣ (2 * m)! * (2 * n)! := by sorry
