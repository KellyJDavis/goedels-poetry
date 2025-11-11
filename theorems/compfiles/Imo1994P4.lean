import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1994, Problem 4

Determine all ordered pairs of positive integers (m, n) such that

            (n³ + 1) / (mn - 1)

is an integer.
-/

namespace Imo1994P4

/- determine -/ abbrev SolutionSet : Set (ℤ × ℤ) := sorry

theorem imo1994_p4 (m n : ℤ) :
    (m, n) ∈ SolutionSet ↔
    0 < m ∧ 0 < n ∧ (m * n - 1) ∣ (n^3 + 1) := sorry


end Imo1994P4
