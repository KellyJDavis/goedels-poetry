import Mathlib.Data.PNat.Basic
import Mathlib.Algebra.BigOperators.Pi
import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2013, Problem 1

Prove that for any pair of positive integers k and n, there exist k positive integers
m₁, m₂, ..., mₖ (not necessarily different) such that

  1 + (2ᵏ - 1)/ n = (1 + 1/m₁) * (1 + 1/m₂) * ... * (1 + 1/mₖ).

-/

namespace Imo2013P1

theorem imo2013_p1 (n : ℕ+) (k : ℕ) :
    ∃ m : ℕ → ℕ+,
      (1 : ℚ) + (2 ^ k - 1) / n = ∏ i ∈ Finset.range k, (1 + 1 / (m i : ℚ)) := sorry


end Imo2013P1
