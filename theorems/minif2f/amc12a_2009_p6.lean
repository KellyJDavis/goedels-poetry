import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Suppose that $P = 2^m$ and $Q = 3^n$. Which of the following is equal to $12^{mn}$ for every pair of integers $(m,n)$?

$\textbf{(A)}\ P^2Q \qquad \textbf{(B)}\ P^nQ^m \qquad \textbf{(C)}\ P^nQ^{2m} \qquad \textbf{(D)}\ P^{2m}Q^n \qquad \textbf{(E)}\ P^{2n}Q^m$ Show that it is \bold{E)} P^{2n} Q^m.-/
theorem amc12a_2009_p6 (m n p q : ℝ) (h₀ : p = 2 ^ m) (h₁ : q = 3 ^ n) :
    p ^ (2 * n) * q ^ m = 12 ^ (m * n) := by sorry
