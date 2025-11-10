import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1998, Problem 3

For any positive integer $n$,
let $d(n)$ denote the number of positive divisors of $n$ (including 1 and $n$ itself).
Determine all positive integers $k$ such that $d(n^2)/d(n) = k$ for some $n$.
-/

namespace Imo1998P3

/- determine -/ abbrev solution_set : Set ℕ := sorry

theorem imo1998_p3 (k : ℕ) :
    k ∈ solution_set ↔
    ∃ n : ℕ,
     (Finset.card (Nat.divisors (n ^ 2))) = k * Finset.card (Nat.divisors n) := sorry


end Imo1998P3
