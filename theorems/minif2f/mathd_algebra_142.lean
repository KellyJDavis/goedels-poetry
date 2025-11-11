import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- A line $\ell$ passes through the points $B(7,-1)$ and $C(-1,7)$.  The equation of this line can be written in the form $y=mx+b$; compute $m+b$. Show that it is 5.-/
theorem mathd_algebra_142 (m b : ℝ) (h₀ : m * 7 + b = -1) (h₁ : m * -1 + b = 7) : m + b = 5 := by sorry
