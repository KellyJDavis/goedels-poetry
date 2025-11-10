import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2003, Problem 6

Let p be a prime number. Prove that there exists a prime number q
such that for every integer n, the number nᵖ - p is not divisible
by q.
-/

namespace Imo2003P6

theorem imo2003_p6 (p : ℕ) (hp : p.Prime) :
    ∃ q : ℕ, q.Prime ∧ ∀ n, ¬((q : ℤ) ∣ (n : ℤ)^p - (p : ℤ)) := sorry


end Imo2003P6
