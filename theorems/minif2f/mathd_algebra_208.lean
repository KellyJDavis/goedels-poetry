import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the value of $\sqrt{1,\!000,\!000} - \sqrt[3]{1,\!000,\!000}$? Show that it is 900.-/
theorem mathd_algebra_208 : Real.sqrt 1000000 - 1000000 ^ ((1 : ℝ) / 3) = 900 := by sorry
