
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 2022, Problem 4

Determine all pairs of primes (p, q) where p - q and pq - q
are both perfect squares.
-/
abbrev solution_set : Set (ℕ × ℕ) := {(3, 2)}

theorem usa2022_p4 (p q : ℕ) :
    (p, q) ∈ solution_set ↔
    p.Prime ∧ q.Prime ∧
    ∃ a, a^2 + q = p ∧ ∃ b, b^2 + q = p * q := by sorry
