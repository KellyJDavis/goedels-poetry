import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2018, Problem 2

Determine all integers n ≥ 3 such that there exist real numbers
a₁, a₂, ..., aₙ satisfying

  aᵢaᵢ₊₁ + 1 = aᵢ₊₂,

where the indices are taken mod n.
-/

namespace Imo2018P2

/- determine -/ abbrev solution_set : Set ℕ := sorry

theorem imo2018_p2 (n : ℕ) :
    n ∈ solution_set ↔
      3 ≤ n ∧
      ∃ a : ZMod n → ℝ, ∀ i, a i * a (i + 1) = a (i + 2) := sorry


end Imo2018P2
