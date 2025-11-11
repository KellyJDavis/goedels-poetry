import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Prove that $\cos{\frac{\pi}{7}}-\cos{\frac{2\pi}{7}}+\cos{\frac{3\pi}{7}}=\frac{1}{2}$.-/
theorem imo_1963_p5 : Real.cos (Real.pi / 7) - Real.cos (2 * Real.pi / 7) + Real.cos (3 * Real.pi / 7) = 1 / 2 := by sorry
