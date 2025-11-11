import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- If $a,b,$ and $c$ are positive [[real number]]s such that $a(b+c) = 152, b(c+a) = 162,$ and $c(a+b) = 170$, then $abc$ is

$\mathrm{(A)}\ 672
\qquad\mathrm{(B)}\ 688
\qquad\mathrm{(C)}\ 704
\qquad\mathrm{(D)}\ 720
\qquad\mathrm{(E)}\ 750$ Show that it is 720.-/
theorem amc12b_2002_p19 (a b c : ℝ) (h₀ : 0 < a ∧ 0 < b ∧ 0 < c) (h₁ : a * (b + c) = 152)
    (h₂ : b * (c + a) = 162) (h₃ : c * (a + b) = 170) : a * b * c = 720 := by sorry
