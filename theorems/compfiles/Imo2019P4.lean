import Mathlib.Data.Nat.Factorial.BigOperators
import Mathlib.Data.Nat.Multiplicity
import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2019, Problem 4

Determine all positive integers n,k that satisfy the equation

  k! = (2ⁿ - 2⁰)(2ⁿ - 2¹) ... (2ⁿ - 2ⁿ⁻¹).
-/

namespace Imo2019P4

open scoped Nat

/- determine -/ abbrev solution_set : Set (ℕ × ℕ) := sorry

theorem imo2018_p2 (n k : ℕ) :
    (n, k) ∈ solution_set ↔
    0 < n ∧ 0 < k ∧
    (k ! : ℤ) = ∏ i ∈ Finset.range n, ((2:ℤ)^n - (2:ℤ)^i) := sorry


end Imo2019P4
