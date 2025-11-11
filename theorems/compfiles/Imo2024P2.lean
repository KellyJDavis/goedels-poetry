import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2024, Problem 2

Determine all pairs (a,b) of positive integers for which there exist positive integers
g and N such that

   gcd(aⁿ + b, bⁿ + a),   n = 1, 2, ...

holds for all integers n ≥ N.
-/

namespace Imo2024P2

/- determine -/ abbrev solutionSet : Set (ℕ+ × ℕ+) := sorry

theorem imo2024_p2 (a b : ℕ+) :
    (a, b) ∈ solutionSet ↔
    ∃ g N : ℕ+,
      ∀ n : ℕ, N ≤ n → Nat.gcd (a^n + b) (b^n + a) = g := sorry

end Imo2024P2
