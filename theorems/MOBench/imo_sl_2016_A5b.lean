
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2016 A5

2. Prove that, for infinitely many $n ∈ ℕ$, there does not exist $a, b ∈ ℕ$
  such that $0 < b ≤ \sqrt{n}$ and $b^2 n ≤ a^2 ≤ b^2 (n + 1)$.
-/
theorem imo_sl_2016_A5b_part2 (N) :
    ∃ n > N, ¬∃ a b, 0 < b ∧ b ≤ n.sqrt ∧ b ^ 2 * n ≤ a ^ 2 ∧ a ^ 2 ≤ b ^ 2 * (n + 1) := by sorry
