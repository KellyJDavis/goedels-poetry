import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- A $3491$ by $3491$ square has its length decreased by $60$ and its width increased by $60$.  By how much does its area change? Show that it is 3600.-/
theorem mathd_algebra_296 : abs ((3491 - 60) * (3491 + 60) - 3491 ^ 2 : ℤ) = 3600 := by sorry
