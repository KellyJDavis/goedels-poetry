
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2021 A2

For any positive integer $n$, prove that
$$ 4 \sum_{i = 1}^n \sum_{j = 1}^n \left\lfloor \frac{ij}{n + 1} \right\rfloor
  ≥ n^2 (n - 1). $$
Determine the equality cases.
-/
/- special open -/ open Finset
abbrev targetSum (n : ℕ) := 4 * ∑ i ∈ range n, ∑ j ∈ range n, (i + 1) * (j + 1) / (n + 1)

theorem imo_sl_2021_A2 (hn : n ≠ 0) : targetSum n = n ^ 2 * (n - 1) ↔ (n + 1).Prime := by sorry
