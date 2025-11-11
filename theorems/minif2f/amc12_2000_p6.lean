import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Two different prime numbers between $4$ and $18$ are chosen. When their sum is subtracted from their product, which of the following numbers could be obtained?

$\textbf{(A)}\ 22 \qquad\textbf{(B)}\ 60 \qquad\textbf{(C)}\ 119 \qquad\textbf{(D)}\ 180 \qquad\textbf{(E)}\ 231$ Show that it is \textbf{(C) }119.-/
theorem amc12_2000_p6 (p q : ℕ) (h₀ : Nat.Prime p ∧ Nat.Prime q) (h₁ : 4 ≤ p ∧ p ≤ 18)
    (h₂ : 4 ≤ q ∧ q ≤ 18) : ↑p * ↑q - (↑p + ↑q) ≠ (194 : ℕ) := by sorry
