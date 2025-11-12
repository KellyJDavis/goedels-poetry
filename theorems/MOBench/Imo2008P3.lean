
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2008, Problem 3
Prove that there exist infinitely many positive integers `n` such that `n^2 + 1` has a prime
divisor which is greater than `2n + √(2n)`.
-/
theorem imo2008_p3 : ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧
    ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ^ 2 + 1 ∧ (p : ℝ) > 2 * (n : ℝ) + Real.sqrt (2 * (n : ℝ)) := by sorry
