
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
Canadian Mathematical Olympiad 1998, Problem 3

Let n be a natural number such that n ≥ 2. Show that

  (1/(n + 1))(1 + 1/3 + ... + 1/(2n - 1)) > (1/n)(1/2 + 1/4 + ... + 1/2n).
-/
theorem canada1998_p3 (n : ℕ) (hn : 2 ≤ n) :
    (1/(n:ℝ)) * ∑ i ∈ Finset.range n, (1/(2 * (i:ℝ) + 2)) <
    (1/((n:ℝ) + 1)) * ∑ i ∈ Finset.range n, (1/(2 * (i:ℝ) + 1)) := by sorry
