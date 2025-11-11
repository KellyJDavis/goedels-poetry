import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the value of the following expression?

$\frac{100^2-7^2}{70^2-11^2} \cdot \frac{(70-11)(70+11)}{(100-7)(100+7)}$

$\textbf{(A) } 1 \qquad \textbf{(B) } \frac{9951}{9950} \qquad \textbf{(C) } \frac{4780}{4779} \qquad \textbf{(D) } \frac{108}{107} \qquad \textbf{(E) } \frac{81}{80} $ Show that it is \textbf{(A) } 1.-/
theorem amc12b_2020_p2 :
    (100 ^ 2 - 7 ^ 2 : ℝ) / (70 ^ 2 - 11 ^ 2) * ((70 - 11) * (70 + 11) / ((100 - 7) * (100 + 7))) =
      1 := by sorry
