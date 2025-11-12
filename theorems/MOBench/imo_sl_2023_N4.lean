
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2023 N4

Let $a_0, a_1, …, a_{n - 1}, b_0, b_1, …, b_{n - 1} ∈ ℕ^+$ and
  $D$ be a positive integer such that for each $i ≤ n$,
$$ b_0 b_1 … b_i a_{i + 1} … a_{n - 1} = b_0 b_1 … b_{i - 1} a_i … a_{n - 1} + D. $$
Determine the smallest possible value of $D$.
-/
/- special open -/ open Finset
structure goodSeq (n : ℕ) where
  a : ℕ → ℕ
  a_pos : ∀ i, 0 < a i
  b : ℕ → ℕ
  b_pos : ∀ i, 0 < b i
  D : ℕ
  D_pos : 0 < D
  spec : ∀ i < n, (range (i + 1)).prod b * (Ico (i + 1) n).prod a
    = (range i).prod b * (Ico i n).prod a + D

theorem imo_sl_2023_N4 (n : ℕ) : IsLeast (Set.range (goodSeq.D (n := n))) n.factorial := by sorry
