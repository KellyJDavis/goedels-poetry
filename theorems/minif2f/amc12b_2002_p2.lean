import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the value of  $(3x - 2)(4x + 1) - (3x - 2)4x + 1$ when $x=4$?

$\mathrm{(A)}\ 0
\qquad\mathrm{(B)}\ 1
\qquad\mathrm{(C)}\ 10
\qquad\mathrm{(D)}\ 11
\qquad\mathrm{(E)}\ 12$ Show that it is \mathrm{(D)}\ 11.-/
theorem amc12b_2002_p2 (x : ℤ) (h₀ : x = 4) :
    (3 * x - 2) * (4 * x + 1) - (3 * x - 2) * (4 * x) + 1 = 11 := by sorry
