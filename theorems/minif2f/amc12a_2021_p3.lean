import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The sum of two natural numbers is $17{,}402$. One of the two numbers is divisible by $10$. If the units digit of that number is erased, the other number is obtained. What is the difference of these two numbers?

$\textbf{(A)} ~10{,}272\qquad\textbf{(B)} ~11{,}700\qquad\textbf{(C)} ~13{,}362\qquad\textbf{(D)} ~14{,}238\qquad\textbf{(E)} Show that it is \textbf{(D)} ~14{,}238.-/
theorem amc12a_2021_p3 (x y : ℕ) (h₀ : x + y = 17402) (h₁ : 10 ∣ x) (h₂ : x / 10 = y) :
    ↑x - ↑y = (14238 : ℤ) := by sorry
