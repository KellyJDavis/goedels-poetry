import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The least common multiple of two numbers is 3720, and their greatest common divisor is 8. Given that one of the numbers is 120, what is the other number? Show that it is 248.-/
theorem mathd_numbertheory_222 (b : ℕ) (h₀ : Nat.lcm 120 b = 3720) (h₁ : Nat.gcd 120 b = 8) :
    b = 248 := by sorry
