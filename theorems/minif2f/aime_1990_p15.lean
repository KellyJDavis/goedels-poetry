import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Find $ax^5 + by^5$ if the real numbers $a,b,x,$ and $y$ satisfy the equations
$\begin{align*}
ax + by &= 3, \\
ax^2 + by^2 &= 7, \\
ax^3 + by^3 &= 16, \\
ax^4 + by^4 &= 42.
\end{align*}$ Show that it is 020.-/
theorem aime_1990_p15 (a b x y : ℝ) (h₀ : a * x + b * y = 3) (h₁ : a * x ^ 2 + b * y ^ 2 = 7)
    (h₂ : a * x ^ 3 + b * y ^ 3 = 16) (h₃ : a * x ^ 4 + b * y ^ 4 = 42) :
    a * x ^ 5 + b * y ^ 5 = 20 := by sorry
