
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2018 C1

Let $n ≥ 4$ be an integer and $S ⊆ ℕ^+$.
We say that $S$ is *good* if for each $m ∈ ℕ$ with $2 ≤ m ≤ n - 2$, there exists
  $T ⊆ S$ of size $m$ such that the sum of all elements in $T$ and $S \ T$ are equal.
Prove that for any $n ≥ 4$, there exists a good set of size $n$.
-/
/- special open -/ open Finset
def good (S : Finset ℕ) :=
  ∀ m ≥ 2, m + 2 ≤ S.card → ∃ T ⊆ S, T.card = m ∧ T.sum id = (S \ T).sum id

theorem imo_sl_2018_C1 (n : ℕ) : ∃ S : Finset ℕ, S.card = n ∧ (∀ x ∈ S, 0 < x) ∧ good S := by sorry
