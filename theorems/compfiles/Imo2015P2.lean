import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2015, Problem 2

Determine all triples of positive integers a, b, c such that each of
ab - c, bc - a, ca - b is a power of two.
-/

namespace Imo2012P2

/- determine -/ abbrev SolutionSet : Set (ℤ × ℤ × ℤ) := sorry

def is_power_of_two (n : ℤ) : Prop := ∃ m : ℕ, n = 2 ^ m

theorem imo2015_p2 (a b c : ℤ) :
    (a,b,c) ∈ SolutionSet ↔
      0 < a ∧ 0 < b ∧ 0 < c ∧
      is_power_of_two (a * b - c) ∧
      is_power_of_two (b * c - a) ∧
      is_power_of_two (c * a - b) := sorry


end Imo2012P2
