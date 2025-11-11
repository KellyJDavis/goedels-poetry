import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Given that $(1+\sin t)(1+\cos t)=5/4$ and
:$(1-\sin t)(1-\cos t)=\frac mn-\sqrt{k},$
where $k, m,$ and $n_{}$ are [[positive integer]]s with $m_{}$ and $n_{}$ [[relatively prime]], find $k+m+n.$ Show that it is 027.-/
theorem aime_1995_p7 (k m n : ℕ) (t : ℝ) (h₀ : 0 < k ∧ 0 < m ∧ 0 < n) (h₁ : Nat.gcd m n = 1)
    (h₂ : (1 + Real.sin t) * (1 + Real.cos t) = 5 / 4)
    (h₃ : (1 - Real.sin t) * (1 - Real.cos t) = m / n - Real.sqrt k) : k + m + n = 27 := by sorry
