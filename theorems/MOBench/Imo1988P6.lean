
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1988, Problem 6

If a and b are two natural numbers such that a*b+1 divides a^2 + b^2,
show that their quotient is a perfect square.
-/
theorem imo1988_p6 {a b : ℕ} (h : a * b + 1 ∣ a ^ 2 + b ^ 2) :
    ∃ d, d ^ 2 = (a ^ 2 + b ^ 2) / (a * b + 1) := by sorry
