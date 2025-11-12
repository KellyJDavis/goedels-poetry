
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2014 N4

Prove that, for any positive integer $n > 1$, there exists infinitely many
  positive integers $k$ such that $⌊n^k/k⌋$ is odd.
-/
theorem imo_sl_2014_N4 (hn : 1 < n) (N) : ∃ k > N, Odd (n ^ k / k) := by sorry
