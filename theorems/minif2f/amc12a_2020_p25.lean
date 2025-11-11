import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The number $a=\frac{p}{q}$, where $p$ and $q$ are relatively prime positive integers, has the property that the sum of all real numbers $x$ satisfying
$ \lfloor x \rfloor \cdot \{x\} = a \cdot x^2$
is $420$, where $\lfloor x \rfloor$ denotes the greatest integer less than or equal to $x$ and $\{x\}=x- \lfloor x \rfloor$ denotes the fractional part of $x$. What is $p+q$?

$\textbf{(A) } 245 \qquad \textbf{(B) } 593 \qquad \textbf{(C) } 929 \qquad \textbf{(D) } 1331 \qquad \textbf{(E) } 1332$ Show that it is \textbf{(C) } 929.-/
theorem amc12a_2020_p25 (a : ℚ) (S : Finset ℝ)
    (h₀ : ∀ x : ℝ, x ∈ S ↔ ↑⌊x⌋ * (x - ↑⌊x⌋) = ↑a * x ^ 2) (h₁ : (∑ k in S, k) = 420) :
    ↑a.den + a.num = 929 := by sorry
