import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Find $n$ if $\gcd(n,40) = 10$ and $\mathop{\text{lcm}}[n,40] = 280$. Show that it is 70.-/
theorem mathd_numbertheory_100 (n : ℕ) (h₀ : 0 < n) (h₁ : Nat.gcd n 40 = 10)
    (h₂ : Nat.lcm n 40 = 280) : n = 70 := by sorry
