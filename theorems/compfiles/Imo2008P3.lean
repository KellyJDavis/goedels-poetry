import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Nat.Prime.Basic
import Mathlib.NumberTheory.PrimesCongruentOne
import Mathlib.NumberTheory.LegendreSymbol.QuadraticReciprocity
import Mathlib.Tactic.LinearCombination

/-!
# International Mathematical Olympiad 2008, Problem 3
Prove that there exist infinitely many positive integers `n` such that `n^2 + 1` has a prime
divisor which is greater than `2n + √(2n)`.
-/

open Real

namespace Imo2008P3

theorem imo2008_p3 : ∀ N : ℕ, ∃ n : ℕ, n ≥ N ∧
    ∃ p : ℕ, Nat.Prime p ∧ p ∣ n ^ 2 + 1 ∧ (p : ℝ) > 2 * n + sqrt (2 * n) := sorry


end Imo2008P3
