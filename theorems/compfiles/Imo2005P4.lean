import Mathlib.FieldTheory.Finite.Basic

/-!
# Intertional Mathematical Olympiad 2005, Problem 4

Determine all positive integers relatively prime to all the terms of the infinite sequence
`a n = 2 ^ n + 3 ^ n + 6 ^ n - 1`, for `n ≥ 1`.
-/

namespace Imo2005P4

def a (n : ℕ) : ℤ := 2 ^ n + 3 ^ n + 6 ^ n - 1

/- determine -/ abbrev SolutionSet : Set ℕ+ := sorry

theorem imo2005_p4 {k : ℕ} (hk : 0 < k) :
    (∀ n : ℕ, 1 ≤ n → IsCoprime (a n) k) ↔ ⟨k, hk⟩ ∈ SolutionSet := sorry


end Imo2005P4
