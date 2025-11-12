
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1964, Problem 1

(a) Find all natural numbers n for which 2ⁿ - 1 is divisible by 7.
-/
abbrev solution_set : Set ℕ := { n | n % 3 = 0 }

theorem imo_1964_p1a (n : ℕ) : n ∈ solution_set ↔ 2^n ≡ 1 [MOD 7] := by sorry
