
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2004, Problem 6

We call a positive integer *alternating* if every two consecutive
digits in its decimal representation are of different parity.

Find all positive integers n such that n has a multiple that is
alternating.
-/
abbrev SolutionSet : Set ℕ :=
  {n : ℕ | 0 < n ∧ ¬(20 ∣ n)}

abbrev Alternating (n : Nat) : Prop :=
  (n.digits 10).Chain' (fun k l ↦ ¬ k ≡ l [MOD 2])

theorem imo2004_p6 (n : ℕ) :
    n ∈ SolutionSet ↔ 0 < n ∧ ∃ k, Alternating (n * k) := by sorry
