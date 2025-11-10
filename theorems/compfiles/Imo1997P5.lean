import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1997, Problem 5

Determine all pairs of integers 1 ≤ a,b that satisfy a ^ (b ^ 2) = b ^ a.
-/

namespace Imo1997P5

/- determine -/ abbrev solution_set : Set (ℕ × ℕ) := sorry

theorem imo1997_p5 (a b : ℕ) (ha : 1 ≤ a) (hb : 1 ≤ b) :
    ⟨a,b⟩ ∈ solution_set ↔ a ^ (b ^ 2) = b ^ a := sorry


end Imo1997P5
