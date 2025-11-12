
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2021 A3

Find the smallest possible value of
$$ \sum_{j = 1}^n \left\lfloor \frac{a_j}{j} \right\rfloor $$
  across all permutations $(a_1, a_2, \ldots, a_n)$ of $(1, 2, \ldots, n)$.
-/
/- special open -/ open List
def targetSum : List ℕ → ℕ
  | [] => 0
  | a :: l => a / (a :: l).length + targetSum l

theorem imo_sl_2021_A3 :
    IsLeast (targetSum '' {l : List ℕ | l ~ (List.range' 1 n).reverse}) n.size := by sorry
