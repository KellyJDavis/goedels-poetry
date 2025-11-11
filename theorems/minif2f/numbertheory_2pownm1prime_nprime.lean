import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Show that if $n$ is a positive integer and $2^n - 1$ is prime, then $n$ is prime.-/
theorem numbertheory_2pownm1prime_nprime (n : ℕ) (h₀ : 0 < n) (h₁ : Nat.Prime (2 ^ n - 1)) : Nat.Prime n := by sorry
