import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Prove that the number $\sum^n_{k=0}\binom{2n+1}{2k+1}2^{3k}$ is not divisible by $5$ for any integer $n\ge0.$-/
theorem imo_1974_p3 (n : ℕ) :
    ¬5 ∣ ∑ k in Finset.range (n + 1), Nat.choose (2 * n + 1) (2 * k + 1) * 2 ^ (3 * k) := by sorry
