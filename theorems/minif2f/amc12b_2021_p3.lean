import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Suppose$2+\frac{1}{1+\frac{1}{2+\frac{2}{3+x}}}=\frac{144}{53}.$What is the value of $x?$

$\textbf{(A) }\frac34 \qquad \textbf{(B) }\frac78 \qquad \textbf{(C) }\frac{14}{15} \qquad \textbf{(D) }\frac{37}{38} \qquad \textbf{(E) }\frac{52}{53}$ Show that it is \text{A}.-/
theorem amc12b_2021_p3 (x : ℝ) (h₀ : 2 + 1 / (1 + 1 / (2 + 2 / (3 + x))) = 144 / 53) : x = 3 / 4 := by sorry
