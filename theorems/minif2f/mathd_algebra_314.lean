import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- If $n = 11$, then what is $\left(\frac{1}{4}\right)^{n+1} \cdot 2^{2n}$? Show that it is \frac{1}{4}.-/
theorem mathd_algebra_314 (n : ℕ) (h₀ : n = 11) : (1 / 4 : ℝ) ^ (n + 1) * 2 ^ (2 * n) = 1 / 4 := by sorry
