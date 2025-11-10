import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1990, Problem 3

Find all integers n > 1 such that n² divides 2ⁿ + 1.
-/

namespace Imo1990P3

/- determine -/ abbrev solution_set : Set ℕ := sorry

theorem imo1990_p3 (n : ℕ) (hn : 1 < n) : n ∈ solution_set ↔ n^2 ∣ 2^n + 1 := sorry


end Imo1990P3
