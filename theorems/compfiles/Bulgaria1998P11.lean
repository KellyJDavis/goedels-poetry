import Mathlib.Tactic

/-!
Bulgarian Mathematical Olympiad 1998, Problem 11

Let m,n be natural numbers such that

   A = ((m + 3)ⁿ + 1) / (3m)

is an integer. Prove that A is odd.
-/

namespace Bulgaria1998P11

theorem bulgaria1998_p11
    (m n A : ℕ)
    (h : 3 * m * A = (m + 3)^n + 1) : Odd A := sorry


end Bulgaria1998P11
