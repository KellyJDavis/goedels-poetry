
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2017, Problem 1

For any integer a₀ > 1, define the sequence

  aₙ₊₁ =   √aₙ, if aₙ is a perfect square
        or aₙ + 3 otherwise.

Find all values of a₀ for which there exists A such that aₙ = A for
infinitely many values of n.
-/
noncomputable def a (a0 : ℕ) : ℕ → ℕ
| 0 => a0
| n + 1 => if (Nat.sqrt (a a0 n))^2 = a a0 n
           then Nat.sqrt (a a0 n)
           else a a0 n + 3

abbrev solution_set : Set ℕ := {n : ℕ | n > 1 ∧ n % 3 = 0}

theorem imo2017_p1 (a0 : ℕ) :
    a0 ∈ solution_set ↔ ∃ A, { n | a a0 n = A }.Infinite := by sorry
