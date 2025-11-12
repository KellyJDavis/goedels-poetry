
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2019, Problem 4

Determine all positive integers n,k that satisfy the equation

  k! = (2ⁿ - 2⁰)(2ⁿ - 2¹) ... (2ⁿ - 2ⁿ⁻¹).
-/
abbrev solution_set : Set (ℕ × ℕ) := { (1,1), (2,3) }

theorem imo2018_p2 (n k : ℕ) :
    (n, k) ∈ solution_set ↔
    0 < n ∧ 0 < k ∧
    (k ! : ℤ) = ∏ i ∈ Finset.range n, ((2:ℤ)^n - (2:ℤ)^i) := by sorry
