
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2018 A2 (P2)

Let $R$ be a totally ordered commutative ring.
Find all periodic sequences $(a_k)_{k ≥ 0}$ such that for any $k ≥ 0$,
$$ a_{k + 2} = a_k a_{k + 1} + 1. $$

Original problem: Find all possible periods of such sequence.
-/
/- special open -/ open Finset
def good [NonAssocSemiring R] (a : ℕ → R) := ∀ k, a (k + 2) = a k * a (k + 1) + 1

variable (R) [NonAssocRing R] (d : Fin 3)

def stdGoodSeq : ℕ → R := λ n ↦ ![2, -1, -1] (n + d)

variable [LinearOrderedCommRing R] {a : ℕ → R} (ha : good a) (hn : a.Periodic (n + 1))

theorem imo_sl_2018_A2 [LinearOrderedCommRing R] {n : ℕ} :
    (∃ a : ℕ → R, good a ∧ a.Periodic (n + 1)) ↔ 3 ∣ n + 1 := by sorry
