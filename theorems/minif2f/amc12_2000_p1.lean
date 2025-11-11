import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- In the year $2001$, the United States will host the [[International Mathematical Olympiad]].  Let $I,M,$ and $O$ be distinct [[positive integer]]s such that the product $I \cdot M \cdot O = 2001 $.  What is the largest possible value of the sum $I + M + O$?

$\textbf{(A)}\ 23 \qquad \textbf{(B)}\ 55 \qquad \textbf{(C)}\ 99 \qquad \textbf{(D)}\ 111 \qquad \textbf{(E)}\ 671$ Show that it is \text{(E)}.-/
theorem amc12_2000_p1 (i m o : ℕ) (h₀ : i ≠ m ∧ m ≠ o ∧ o ≠ i) (h₁ : i * m * o = 2001) :
    i + m + o ≤ 671 := by sorry
