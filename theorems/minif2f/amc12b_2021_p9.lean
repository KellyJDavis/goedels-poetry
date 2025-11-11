import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the value of$\frac{\log_2 80}{\log_{40}2}-\frac{\log_2 160}{\log_{20}2}?$$\textbf{(A) }0 \qquad \textbf{(B) }1 \qquad \textbf{(C) }\frac54 \qquad \textbf{(D) }2 \qquad \textbf{(E) }\log_2 5$ Show that it is \text{(D)}.-/
theorem amc12b_2021_p9 :
    Real.log 80 / Real.log 2 / (Real.log 2 / Real.log 40) -
        Real.log 160 / Real.log 2 / (Real.log 2 / Real.log 20) =
      2 := by sorry
