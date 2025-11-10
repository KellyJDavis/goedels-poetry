import Mathlib.Algebra.IsPrimePow
import Mathlib.Data.Nat.ModEq
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.Tactic.Common

/-!
# International Mathematical Olympiad 1989, Problem 5

Prove that for each positive integer n there exist n consecutive positive
integers, none of which is an integral power of a prime number.
-/

namespace Imo1989P5

theorem imo1989_p5 (n : ℕ) : ∃ m, ∀ j < n, ¬IsPrimePow (m + j) := sorry


end Imo1989P5
