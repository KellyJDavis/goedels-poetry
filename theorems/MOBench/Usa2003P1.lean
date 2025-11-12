
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 2003, Problem 1

Prove that for every positive integer n there exists an n-digit
number divisible by 5ⁿ, all of whose digits are odd.
-/
theorem usa2003_p1 (n : ℕ) :
    ∃ m, (Nat.digits 10 m).length = n ∧
          5^n ∣ m ∧ (Nat.digits 10 m).all (Odd ·) := by sorry
