
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2015, Problem 2

Determine all triples of positive integers a, b, c such that each of
ab - c, bc - a, ca - b is a power of two.
-/
abbrev SolutionSet : Set (ℤ × ℤ × ℤ) :=
  {(2, 2, 2), (2, 2, 3), (3, 2, 2), (2, 3, 2),
   (2, 6, 11), (2, 11, 6), (6, 2, 11), (6, 11, 2), (11, 2, 6), (11, 6, 2),
   (3, 5, 7), (3, 7, 5), (5, 3, 7), (5, 7, 3), (7, 3, 5), (7, 5, 3)}

def is_power_of_two (n : ℤ) : Prop := ∃ m : ℕ, n = 2 ^ m

theorem imo2015_p2 (a b c : ℤ) :
    (a,b,c) ∈ SolutionSet ↔
      0 < a ∧ 0 < b ∧ 0 < c ∧
      is_power_of_two (a * b - c) ∧
      is_power_of_two (b * c - a) ∧
      is_power_of_two (c * a - b) := by sorry
