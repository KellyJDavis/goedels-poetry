import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- In the complex plane, let $A$ be the set of solutions to $z^{3}-8=0$ and let $B$ be the set of solutions to $z^{3}-8z^{2}-8z+64=0.$ What is the greatest distance between a point of $A$ and a point of $B?$

$\textbf{(A) } 2\sqrt{3} \qquad \textbf{(B) } 6 \qquad \textbf{(C) } 9 \qquad \textbf{(D) } 2\sqrt{21} \qquad \textbf{(E) } 9+\sqrt{3}$ Show that it is \textbf{(D) } 2\sqrt{21}.-/
theorem amc12a_2020_p15 (a b : ℂ) (h₀ : a ^ 3 - 8 = 0) (h₁ : b ^ 3 - 8 * b ^ 2 - 8 * b + 64 = 0) :
    Complex.abs (a - b) ≤ 2 * Real.sqrt 21 := by sorry
