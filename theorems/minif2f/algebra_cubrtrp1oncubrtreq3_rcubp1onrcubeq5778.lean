import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Let $r$ be a real number such that $r^{\frac{1}{3}} + \frac{1}{r^{\frac{1}{3}}} = 3$. Show that $r^3 + \frac{1}{r^3} = 5778$.-/
theorem algebra_cubrtrp1oncubrtreq3_rcubp1onrcubeq5778 (r : ℝ) (hr : r ≥ 0)
    (h₀ : r ^ ((1 : ℝ) / 3) + 1 / r ^ ((1 : ℝ) / 3) = 3) : r ^ 3 + 1 / r ^ 3 = 5778 := by sorry
