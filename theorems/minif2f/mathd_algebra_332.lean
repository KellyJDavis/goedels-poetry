import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Real numbers $x$ and $y$ have an arithmetic mean of 7 and a geometric mean of $\sqrt{19}$. Find $x^2+y^2$. Show that it is 158.-/
theorem mathd_algebra_332 (x y : ℝ) (h₀ : (x + y) / 2 = 7) (h₁ : Real.sqrt (x * y) = Real.sqrt 19) :
    x ^ 2 + y ^ 2 = 158 := by sorry
