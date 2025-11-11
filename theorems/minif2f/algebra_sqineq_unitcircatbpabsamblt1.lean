import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Let $a$ and $b$ be real numbers such that $a^2+b^2=1$. Show that $ab+\lvert a-b\rvert \leq 1$.-/
theorem algebra_sqineq_unitcircatbpabsamblt1 (a b : ℝ) (h₀ : a ^ 2 + b ^ 2 = 1) :
    a * b + abs (a - b) ≤ 1 := by sorry
