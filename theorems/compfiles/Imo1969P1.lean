import Mathlib.Algebra.Ring.Identities
import Mathlib.Data.Int.NatPrime
import Mathlib.Tactic.Linarith
import Mathlib.Data.Set.Finite.Basic

/-!
# International Mathematical Olympiad 1969, Problem 1

Prove that there are infinitely many natural numbers a with the following property:
the number z = n⁴ + a is not prime for any natural number n.
-/

open Int Nat

namespace Imo1969P1

theorem imo1969_p1 : Set.Infinite {a : ℕ | ∀ n : ℕ, ¬Nat.Prime (n ^ 4 + a)} := sorry


end Imo1969P1
