import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Suppose $f(x)$ is an invertible function, and suppose that $f(2)=f^{-1}(2)=4$.

What is the value of $f(f(2))$? Show that it is 2.-/
theorem mathd_algebra_188 (σ : Equiv ℝ ℝ) (h : σ.1 2 = σ.2 2) : σ.1 (σ.1 2) = 2 := by sorry
