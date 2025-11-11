import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the value of $\left(\sum_{k=1}^{20} \log_{5^k} 3^{k^2}\right)\cdot\left(\sum_{k=1}^{100} \log_{9^k} 25^k\right)?$

$\textbf{(A) }21 \qquad \textbf{(B) }100\log_5 3 \qquad \textbf{(C) }200\log_3 5 \qquad \textbf{(D) }2{,}200\qquad \textbf{(E) }21{,}000$ Show that it is \textbf{(E) }21{,}000.-/
theorem amc12a_2021_p14 :
    ((∑ k in Finset.Icc 1 20, Real.logb (5 ^ k) (3 ^ k ^ 2)) *
        ∑ k in Finset.Icc 1 100, Real.logb (9 ^ k) (25 ^ k)) = 21000 := by sorry
