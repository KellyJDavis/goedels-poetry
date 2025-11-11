import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Five plus $500\%$ of $10$ is the same as $110\%$ of what number? Show that it is 50.-/
theorem mathd_algebra_400 (x : ℝ) (h₀ : 5 + 500 / 100 * 10 = 110 / 100 * x) : x = 50 := by sorry
