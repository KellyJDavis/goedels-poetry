
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1990, Problem 3

Find all integers n > 1 such that n² divides 2ⁿ + 1.
-/
abbrev solution_set : Set ℕ := { 3 }

theorem imo1990_p3 (n : ℕ) (hn : 1 < n) : n ∈ solution_set ↔ n^2 ∣ 2^n + 1 := by sorry
