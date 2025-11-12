
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2021 A6 (P6)

Let $m ∈ ℕ$ and $a_0, a_1, …, a_{k - 1}$ be integers.
Suppose that there exists subsets $B_0, B_1, …, B_{m - 1}$ of $[k]$
  such that for each $i ∈ [m]$, $$ \sum_{j ∈ B_i} a_j = m^{i + 1}. $$
Prove that $k ≥ m/2$.
-/
/- special open -/ open Finset
variable [Fintype κ] [DecidableEq κ] {a : κ → ℤ}

theorem imo_sl_2021_A6 {a : κ → ℤ} {B : Fin m → Finset κ}
    [∀ i j, Decidable (j ∈ B i)] (h : ∀ i : Fin m, (B i).sum a = m ^ (i.1 + 1)) :
    m ≤ 2 * Fintype.card κ := by sorry
