import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Suppose $a$ and $b$ are positive integers such that the units digit of $a$ is $2$, the units digit of $b$ is $4$, and the greatest common divisor of $a$ and $b$ is $6$.

What is the smallest possible value of the least common multiple of $a$ and $b$? Show that it is 108.-/
theorem mathd_numbertheory_495 (a b : ℕ) (h₀ : 0 < a ∧ 0 < b) (h₁ : a % 10 = 2) (h₂ : b % 10 = 4)
    (h₃ : Nat.gcd a b = 6) : 108 ≤ Nat.lcm a b := by sorry
