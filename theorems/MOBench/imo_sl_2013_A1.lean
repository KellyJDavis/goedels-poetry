
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2013 A1

Let $R$ be a commutative ring.
Given a list of elements $a_0, \dots, a_{n-1} \in R$, we define a sequence $(u_k)$ by
$u_0 = u_1 = 1$, and $u_{k + 2} = u_{k + 1} + a_k u_k$ for each $0 \le k < n$.
We then define the function $f(a_0, \dots, a_{n-1}) = u_{n + 1}$.

Prove that $f(a_0, \dots, a_{n-1}) = f(a_{n-1}, \dots, a_0)$.
-/
variable {R : Type*} [CommRing R]

/--
A helper function to compute the pair `(u_{k+1}, u_k)` recursively.
`f_aux [a₀, a₁, ..., a_{k-1}]` returns `(u_{k+1}, u_k)`.
-/
def f_aux : List R → R × R
  | [] => (1, 1)
  | r :: l => let (a, b) := f_aux l; (a + r * b, a)

def f (l : List R) : R := (f_aux l).1

theorem imo_sl_2013_A1 (l : List R) : f l.reverse = f l := by sorry
