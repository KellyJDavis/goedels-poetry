import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Let $A, M,$ and $C$ be [[nonnegative integer]]s such that $A + M + C=12$. What is the maximum value of $A \cdot M \cdot C + A \cdot M + M \cdot C + A \cdot C$?

$ \mathrm{(A) \ 62 } \qquad \mathrm{(B) \ 72 } \qquad \mathrm{(C) \ 92 } \qquad \mathrm{(D) \ 102 } \qquad \mathrm{(E) \ 112 }  $ Show that it is \text{E}.-/
theorem amc12_2000_p12 (a m c : ℕ) (h₀ : a + m + c = 12) :
    a * m * c + a * m + m * c + a * c ≤ 112 := by sorry
