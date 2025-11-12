
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2016 A5

1. Prove that, for every $n ∈ ℕ$, there exists some $a, b ∈ ℕ$
  such that $0 < b ≤ \sqrt{n} + 1$ and $b^2 n ≤ a^2 ≤ b^2 (n + 1)$.
-/
theorem imo_sl_2016_A5a_part1 (n) :
    ∃ a b, 0 < b ∧ b ≤ n.sqrt + 1 ∧ b ^ 2 * n ≤ a ^ 2 ∧ a ^ 2 ≤ b ^ 2 * (n + 1) := by sorry
