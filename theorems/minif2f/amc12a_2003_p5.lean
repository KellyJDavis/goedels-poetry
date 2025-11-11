import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The sum of the two 5-digit numbers $AMC10$ and $AMC12$ is $123422$. What is $A+M+C$?

$ \mathrm{(A) \ } 10\qquad \mathrm{(B) \ } 11\qquad \mathrm{(C) \ } 12\qquad \mathrm{(D) \ } 13\qquad \mathrm{(E) \ } 14 $ Show that it is \mathrm{(E)}\ 14.-/
theorem amc12a_2003_p5 (A M C : ℕ) (h₀ : A ≤ 9 ∧ M ≤ 9 ∧ C ≤ 9)
    (h₁ : Nat.ofDigits 10 [0, 1, C, M, A] + Nat.ofDigits 10 [2, 1, C, M, A] = 123422) :
    A + M + C = 14 := by sorry
