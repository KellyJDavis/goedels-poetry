
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1974, Problem 3

Prove that the sum from k = 0 to n inclusive of
   Choose[2n + 1, 2k + 1] * 2³ᵏ
is not divisible by 5 for any integer n ≥ 0.
-/
theorem imo1974_p3
    (n : ℕ) :
    ¬ 5 ∣ ∑ k ∈ Finset.range (n + 1),
            (Nat.choose (2 * n + 1) (2 * k + 1)) * (2^(3 * k)) := by sorry
