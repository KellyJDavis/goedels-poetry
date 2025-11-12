
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2014 C4

Consider 4 types of skew-tetrominoes in $ℕ^2$, classified by its orientation.
Let $S ⊆ ℕ^2$ be a multiset, and suppose that it can be partitioned into skew-tetrominos.
Prove that the parity of the number of skew-tetrominoes used for
  each type in the partition does not depend on the partition.
-/
/- special open -/ open Multiset
/-- Base skew-tetrominoes, representing the four orientations. -/
def BaseSkewT4 : Fin 4 → Multiset (ℕ × ℕ)
  | 1 => {(0, 0), (1, 0), (1, 1), (2, 1)}
  | 2 => {(1, 0), (1, 1), (0, 1), (0, 2)}
  | 3 => {(0, 1), (1, 1), (1, 0), (2, 0)}
  | 4 => {(0, 0), (0, 1), (1, 1), (1, 2)}

/-- A specific skew-tetromino piece, defined by its type and position. -/
def SkewT4 (q : Fin 4 × ℕ × ℕ) : Multiset (ℕ × ℕ) :=
  (BaseSkewT4 q.1).map λ p ↦ q.2 + p

/--
Let `P` and `Q` be two different partitions of the same shape `S` into skew-tetrominoes.
This is formally stated as `(map SkewT4 P).sum = (map SkewT4 Q).sum`.
The theorem asserts that for any type `i`, the number of tetrominoes of that type
has the same parity in both partitions.
-/
theorem imo_sl_2014_C4 {P Q : Multiset (Fin 4 × ℕ × ℕ)}
    (h : (map SkewT4 P).sum = (map SkewT4 Q).sum) (i : Fin 4) :
    (P.map Prod.fst).count i % 2 = (Q.map Prod.fst).count i % 2 := by sorry
