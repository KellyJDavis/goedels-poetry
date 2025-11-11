import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Two different positive numbers $a$ and $b$ each differ from their reciprocals by $1$. What is $a+b$?

$
\text{(A) }1
\qquad
\text{(B) }2
\qquad
\text{(C) }\sqrt 5
\qquad
\text{(D) }\sqrt 6
\qquad
\text{(E) }3
$ Show that it is (C) \sqrt 5.-/
theorem amc12a_2002_p13 (a b : ℝ) (h₀ : 0 < a ∧ 0 < b) (h₁ : a ≠ b) (h₂ : abs (a - 1 / a) = 1)
    (h₃ : abs (b - 1 / b) = 1) : a + b = Real.sqrt 5 := by sorry
