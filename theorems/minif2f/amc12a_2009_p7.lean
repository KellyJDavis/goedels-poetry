import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The first three terms of an arithmetic sequence are $2x - 3$, $5x - 11$, and $3x + 1$ respectively. The $n$th term of the sequence is $2009$. What is $n$?

$\textbf{(A)}\ 255 \qquad \textbf{(B)}\ 502 \qquad \textbf{(C)}\ 1004 \qquad \textbf{(D)}\ 1506 \qquad \textbf{(E)}\ 8037$ Show that it is 502.-/
theorem amc12a_2009_p7 (x : ℝ) (n : ℕ) (a : ℕ → ℝ)
    (h₁ : ∀ m, a (m + 1) - a m = a (m + 2) - a (m + 1)) (h₂ : a 1 = 2 * x - 3)
    (h₃ : a 2 = 5 * x - 11) (h₄ : a 3 = 3 * x + 1) (h₅ : a n = 2009) : n = 502 := by sorry
