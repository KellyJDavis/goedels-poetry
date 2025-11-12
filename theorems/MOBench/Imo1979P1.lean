
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1979, Problem 1

Suppose that p and q are positive integers such that

  p / q = 1 - 1/2 + 1/3 - 1/4 + ... - 1/1318 + 1/1319.

Prove that p is divisible by 1979.
-/
theorem imo1979_p1 (p q : ℤ) (hp : 0 < p) (hq : 0 < q)
    (h : (p : ℚ) / q = ∑ i ∈ Finset.range 1319, (-1 : ℚ)^i / (i + 1)) :
    1979 ∣ p := by sorry
