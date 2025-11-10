import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1985, Problem 4

Given a set M of 1985 distinct positive integers, none of which has a prime
divisor greater than 23, prove that M contains a subset of 4 elements
whose product is the 4th power of an integer.
-/

namespace Imo1985P4

theorem imo1985_p4 (M : Finset ℕ) (Mpos : ∀ m ∈ M, 0 < m)
    (Mdivisors : ∀ m ∈ M, ∀ n, m.Prime ∧ n ∣ m → m ≤ 23)
    : ∃ M' : Finset ℕ, M' ⊆ M ∧ ∃ k, M'.prod id = k^4 := sorry


end Imo1985P4
