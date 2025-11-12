
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1971, Problem 3

Prove that we can find an infinite set of positive integers of the form 2^n - 3
(where n is a positive integer) every pair of which are relatively prime.
-/
theorem imo1971_p3 : Set.Infinite
  {(n, m) : ℕ × ℕ | Nat.Coprime (2 ^ n - 3) (2 ^ m - 3)} := by sorry
