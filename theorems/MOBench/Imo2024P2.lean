
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2024, Problem 2

Determine all pairs (a,b) of positive integers for which there exist positive integers
g and N such that

   gcd(aⁿ + b, bⁿ + a),   n = 1, 2, ...

holds for all integers n ≥ N.
-/
abbrev solutionSet : Set (ℕ+ × ℕ+) := {(1, 1)}

theorem imo2024_p2 (a b : ℕ+) :
    (a, b) ∈ solutionSet ↔
    ∃ g N : ℕ+,
      ∀ n : ℕ, N ≤ n → Nat.gcd (a^n + b) (b^n + a) = g := by sorry
