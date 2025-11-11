import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The product of three consecutive positive integers is $8$ times their sum. What is the sum of their [[perfect square|squares]]?

$\mathrm{(A)}\ 50
\qquad\mathrm{(B)}\ 77
\qquad\mathrm{(C)}\ 110
\qquad\mathrm{(D)}\ 149
\qquad\mathrm{(E)}\ 194$ Show that it is \mathrm{ (B)}\ 77.-/
theorem amc12b_2002_p7 (a b c : ℕ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c) (h₁ : b = a + 1) (h₂ : c = b + 1)
    (h₃ : a * b * c = 8 * (a + b + c)) : a ^ 2 + (b ^ 2 + c ^ 2) = 77 := by sorry
