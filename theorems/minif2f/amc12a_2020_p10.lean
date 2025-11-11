import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- There is a unique positive integer $n$ such that$\log_2{(\log_{16}{n})} = \log_4{(\log_4{n})}.$What is the sum of the digits of $n?$

$\textbf{(A) } 4 \qquad \textbf{(B) } 7 \qquad \textbf{(C) } 8 \qquad \textbf{(D) } 11 \qquad \textbf{(E) } 13$ Show that it is \textbf{(E) } 13.-/
theorem amc12a_2020_p10 (n : ℕ) (h₀ : 1 < n)
    (h₁ : Real.logb 2 (Real.logb 16 n) = Real.logb 4 (Real.logb 4 n)) :
    (List.sum (Nat.digits 10 n)) = 13 := by sorry
