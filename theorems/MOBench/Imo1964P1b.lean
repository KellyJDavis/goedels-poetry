
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1964, Problem 1

(b) Prove that there is no positive integer n for which 2ⁿ + 1 is divisible by 7.
-/
theorem imo_1964_p1b (n : ℕ) : ¬ 7 ∣ (2^n + 1) := by sorry
