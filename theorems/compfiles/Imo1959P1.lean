import Mathlib.Algebra.Divisibility.Basic
import Mathlib.Tactic.Ring
import Mathlib.Data.Nat.Prime.Basic

/-!
# International Mathematical Olympiad 1959, Problem 1.

Prove that the fraction `(21n+4)/(14n+3)` is irreducible for every
natural number `n`.
-/

namespace Imo1959P1

/-
Since Lean doesn't have a concept of "irreducible fractions" per se,
we just formalize this as saying the numerator and denominator are
relatively prime.
-/
theorem imo1959_p1 : ∀ n : ℕ, Nat.Coprime (21 * n + 4) (14 * n + 3) := sorry


end Imo1959P1
