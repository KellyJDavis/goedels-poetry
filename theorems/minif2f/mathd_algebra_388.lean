import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- If

\begin{align*}
3x+4y-12z&=10,\\
-2x-3y+9z&=-4,
\end{align*}

compute $x$. Show that it is 14.-/
theorem mathd_algebra_388 (x y z : ℝ) (h₀ : 3 * x + 4 * y - 12 * z = 10)
    (h₁ : -2 * x - 3 * y + 9 * z = -4) : x = 14 := by sorry
