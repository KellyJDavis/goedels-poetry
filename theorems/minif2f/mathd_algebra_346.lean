import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Let $f(x) = 2x-3$ and $g(x) = x+1$. What is the value of $g(f(5)-1)$? Show that it is 7.-/
theorem mathd_algebra_346 (f g : ℝ → ℝ) (h₀ : ∀ x, f x = 2 * x - 3) (h₁ : ∀ x, g x = x + 1) :
    g (f 5 - 1) = 7 := by sorry
