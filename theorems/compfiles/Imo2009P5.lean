import Mathlib.Algebra.Order.Positive.Field
import Mathlib.Data.Nat.Periodic
import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2009, Problem 5

Determine all functions f: ℤ>0 → ℤ>0 such that for all positive integers a and b,
the numbers

  a, f(b), and f(b + f(a) - 1)

form the sides of a nondegenerate triangle.
-/

namespace Imo2009P5

/- determine -/ abbrev solution_set : Set (ℕ+ → ℕ+) := sorry

theorem imo2009_p5 (f : ℕ+ → ℕ+) :
    f ∈ solution_set ↔
    ∀ a b, (f (b + f a - 1) < f b + a ∧
            a < f b + f (b + f a - 1) ∧
            f b < f (b + f a - 1) + a) := sorry


end Imo2009P5
