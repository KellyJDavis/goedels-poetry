import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The greatest common divisor of positive integers $m$ and $n$ is 8. The least common multiple of $m$ and $n$ is 112. What is the least possible value of $m+n$? Show that it is 72.-/
theorem mathd_numbertheory_711 (m n : ℕ) (h₀ : 0 < m ∧ 0 < n) (h₁ : Nat.gcd m n = 8)
    (h₂ : Nat.lcm m n = 112) : 72 ≤ m + n := by sorry
