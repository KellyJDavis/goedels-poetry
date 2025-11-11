import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 2022, Problem 4

Determine all pairs of primes (p, q) where p - q and pq - q
are both perfect squares.
-/

namespace Usa2022P4

/- determine -/ abbrev solution_set : Set (ℕ × ℕ) := sorry

theorem usa2022_p4 (p q : ℕ) :
    (p, q) ∈ solution_set ↔
    p.Prime ∧ q.Prime ∧
    ∃ a, a^2 + q = p ∧ ∃ b, b^2 + q = p * q := sorry


end Usa2022P4
