import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Which of the following is equivalent to
$(2+3)(2^2+3^2)(2^4+3^4)(2^8+3^8)(2^{16}+3^{16})(2^{32}+3^{32})(2^{64}+3^{64})?$
$\textbf{(A)} ~3^{127} + 2^{127} \qquad\textbf{(B)} ~3^{127} + 2^{127} + 2 \cdot 3^{63} + 3 \cdot 2^{63} \qquad\textbf{(C)} ~3^{128}-2^{128} \qquad\textbf{(D)} ~3^{128} + 2^{128} \qquad\textbf{(E)} Show that it is \textbf{(C)} ~3^{128}-2^{128}.-/
theorem amc12a_2021_p9 : (∏ k in Finset.range 7, (2 ^ 2 ^ k + 3 ^ 2 ^ k)) = 3 ^ 128 - 2 ^ 128 := by sorry
