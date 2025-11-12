
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2011 A1

Consider an arbitrary set $A = \{a_1, a_2, a_3, a_4\}$ of four distinct positive integers.
Let $p_A$ be the number of pairs $(i, j)$ with $1 \le i < j \le 4$
such that $a_i + a_j$ divides $a_1 + a_2 + a_3 + a_4$.
Determine all sets $A$ of size $4$ such that $p_A \ge p_B$ for all sets $B$ of size $4$.
-/
/- special open -/ open Finset
/--
A `Card4NatSet` represents a set of four distinct positive integers,
formalized as a strictly increasing sequence of length 4.
-/
@[ext] structure Card4NatSet where
  f : Fin 4 → ℕ
  f_pos : ∀ i, 0 < f i
  f_strict_mono : StrictMono f

/--
`p_val A` is the number $p_A$ from the problem statement.
It counts the pairs `(i, j)` with `i < j` such that `aᵢ + aⱼ` divides the total sum.
-/
def p_val (A : Card4NatSet) : ℕ :=
  let S := A.f 0 + A.f 1 + A.f 2 + A.f 3
  (univ.filter fun (p : Fin 4 × Fin 4) ↦ p.1 < p.2 ∧ A.f p.1 + A.f p.2 ∣ S).card

/--
The main theorem characterizes the sets `A` which maximize `p_val`.
The solutions are precisely the positive integer multiples of the sets
$\{1, 5, 7, 11\}$ and $\{1, 11, 19, 29\}$.
-/
theorem imo_sl_2011_A1 (A : Card4NatSet) :
  (∀ B : Card4NatSet, p_val B ≤ p_val A) ↔
    (∃ (n : ℕ) (_ : 0 < n), A.f = n • ![1, 5, 7, 11] ∨ A.f = n • ![1, 11, 19, 29]) := by sorry
