
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1997, Problem 5

Determine all pairs of integers 1 ≤ a,b that satisfy a ^ (b ^ 2) = b ^ a.
-/
abbrev solution_set : Set (ℕ × ℕ) := {(1, 1), (16, 2), (27, 3)}

theorem imo1997_p5 (a b : ℕ) (ha : 1 ≤ a) (hb : 1 ≤ b) :
    ⟨a,b⟩ ∈ solution_set ↔ a ^ (b ^ 2) = b ^ a := by sorry
