import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 2008, Problem 1

Prove that for each positive integer n, there are pairwise relatively prime
integers k₀,k₁,...,kₙ, all strictly greater than 1, such that k₀k₁...kₙ - 1
is a product of two consecutive integers.
-/

namespace Usa2008P1

theorem usa2008_p1 (n : ℕ) (hn : 0 < n) :
    ∃ k : Fin (n + 1) → ℕ,
      (∀ i, 1 < k i) ∧
      (∀ i j, i ≠ j → Nat.Coprime (k i) (k j)) ∧
      ∃ m, ∏ i : Fin (n + 1), k i = 1 + m * (m + 1) := sorry


end Usa2008P1
