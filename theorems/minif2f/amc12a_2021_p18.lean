import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Let $f$ be a function defined on the set of positive rational numbers with the property that $f(a\cdot b)=f(a)+f(b)$ for all positive rational numbers $a$ and $b$. Suppose that $f$ also has the property that $f(p)=p$ for every prime number $p$. For which of the following numbers $x$ is $f(x)<0$?

$\textbf{(A) }\frac{17}{32} \qquad \textbf{(B) }\frac{11}{16} \qquad \textbf{(C) }\frac79 \qquad \textbf{(D) }\frac76\qquad \textbf{(E) }\frac{25}{11}$ Show that it is \textbf{(E) }\frac{25}{11}.-/
theorem amc12a_2021_p18 (f : ℚ → ℝ) (h₀ : ∀ x > 0, ∀ y > 0, f (x * y) = f x + f y)
    (h₁ : ∀ p, Nat.Prime p → f p = p) : f (25 / 11) < 0 := by sorry
