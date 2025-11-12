
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2006, Problem 5

Let $P(x)$ be a polynomial of degree $n>1$ with integer coefficients, and let $k$ be a positive
integer. Consider the polynomial $Q(x) = P(P(\ldots P(P(x))\ldots))$, where $P$ occurs $k$ times.
Prove that there are at most $n$ integers $t$ such that $Q(t)=t$.
-/
/- special open -/ open Function Polynomial






theorem imo2006_p5 {P : Polynomial ℤ} (hP : 1 < P.natDegree) {k : ℕ} (hk : 0 < k) :
    (P.comp^[k] X - X).roots.toFinset.card ≤ P.natDegree := by sorry
