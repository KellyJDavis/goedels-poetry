
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2009, Problem 5

Determine all functions f: ℤ>0 → ℤ>0 such that for all positive integers a and b,
the numbers

  a, f(b), and f(b + f(a) - 1)

form the sides of a nondegenerate triangle.
-/
abbrev solution_set : Set (ℕ+ → ℕ+) := { id }

theorem imo2009_p5 (f : ℕ+ → ℕ+) :
    f ∈ solution_set ↔
    ∀ a b, (f (b + f a - 1) < f b + a ∧
            a < f b + f (b + f a - 1) ∧
            f b < f (b + f a - 1) + a) := by sorry
