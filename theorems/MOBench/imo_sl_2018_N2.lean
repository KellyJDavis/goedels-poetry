
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2018 N2

Let $n$ and $k$ be positive integers.
Consider an $k × k$ table, where each cell contains an integer $1 \pmod{n}$.
Suppose that the sum of all numbers in an arbitrary row or column is $k \pmod{n^2}$.
For each $i ≤ n$, let $R_i$ and $C_i$ be the product of
  numbers in the $i$th row and $i$th column, respectively.
Prove that
$$ \sum_{i = 1}^n R_i ≡ \sum_{i = 1}^n C_i \pmod{n^4}. $$
-/
/- special open -/ open Finset
theorem imo_sl_2018_N2 {n : ℤ} {f : ι → ι → ℤ} (h : ∀ i ∈ S, ∀ j ∈ S, f i j ≡ 1 [ZMOD n])
    (h0 : ∀ i ∈ S, ∑ j ∈ S, f i j ≡ S.card [ZMOD n ^ 2])
    (h1 : ∀ j ∈ S, ∑ i ∈ S, f i j ≡ S.card [ZMOD n ^ 2]) :
    (S.sum λ i ↦ ∏ j ∈ S, f i j) ≡ (S.sum λ j ↦ ∏ i ∈ S, f i j) [ZMOD n ^ 4] := by sorry
