import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the smallest positive integer $N$ such that the value $7 + (30 \times N)$ is not a prime number? Show that it is 6.-/
theorem mathd_numbertheory_150 (n : ℕ) (h₀ : ¬Nat.Prime (7 + 30 * n)) : 6 ≤ n := by sorry
