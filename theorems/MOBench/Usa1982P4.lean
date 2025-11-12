
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 1982, Problem 4

Prove that there exists a positive integer k such that
k⬝2ⁿ + 1 is composite for every integer n.
-/
theorem usa1982_p4 :
    ∃ k : ℕ, 0 < k ∧ ∀ n : ℕ, ¬ Nat.Prime (k * (2 ^ n) + 1) := by sorry
