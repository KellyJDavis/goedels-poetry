import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The [[function]] f is defined on the [[set]] of [[integer]]s and satisfies $f(n)=\begin{cases}
n-3&\mbox{if}\ n\ge 1000\\
f(f(n+5))&\mbox{if}\ n<1000\end{cases}$

Find $f(84)$. Show that it is 997.-/
theorem aime_1984_p7 (f : ℤ → ℤ) (h₀ : ∀ n, 1000 ≤ n → f n = n - 3)
    (h₁ : ∀ n, n < 1000 → f n = f (f (n + 5))) : f 84 = 997 := by sorry
