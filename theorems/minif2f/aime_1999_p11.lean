import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Given that $\sum_{k=1}^{35}\sin 5k=\tan \frac mn,$ where angles are measured in degrees, and $m_{}$ and $n_{}$ are relatively prime positive integers that satisfy $\frac mn<90,$ find $m+n.$ Show that it is 177.-/
theorem aime_1999_p11 (m : ℚ) (h₀ : 0 < m)
    (h₁ : (∑ k in Finset.Icc (1 : ℕ) 35, Real.sin (5 * k * Real.pi / 180)) = Real.tan (m * Real.pi / 180))
    (h₂ : (m.num : ℝ) / m.den < 90) : ↑m.den + m.num = 177 := by sorry
