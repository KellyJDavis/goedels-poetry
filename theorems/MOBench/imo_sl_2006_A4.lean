
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2006 A4

Let $F$ be a totally ordered field and $a_1, a_2, …, a_n ∈ F$ be positive.
Prove the inequality
$$ \sum_{i < j} \frac{a_i a_j}{a_i + a_j}
  ≤ \frac{n}{2(a_1 + a_2 + … + a_n)} \sum_{i < j} a_i a_j. $$
-/
theorem imo_sl_2006_A4 [LinearOrderedField F] [LinearOrder ι]
    (a : ι → F) {S : Finset ι} (hS : ∀ i ∈ S, 0 < a i) :
    let T := (S ×ˢ S).filter λ p ↦ p.1 < p.2
    T.sum (λ p ↦ a p.1 * a p.2 / (a p.1 + a p.2))
      ≤ S.card • T.sum (λ p ↦ a p.1 * a p.2) / (2 * S.sum a) := by sorry
