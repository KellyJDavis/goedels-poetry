
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1989, Problem 5

Prove that for each positive integer n there exist n consecutive positive
integers, none of which is an integral power of a prime number.
-/
theorem imo1989_p5 (n : ℕ) : ∃ m, ∀ j < n, ¬IsPrimePow (m + j) := by sorry
