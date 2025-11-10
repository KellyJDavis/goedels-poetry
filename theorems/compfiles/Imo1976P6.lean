import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1976, Problem 6

The sequence u_0, u_1, u_2, ... is defined by:
u_0 = 2, u1 = 5/2, u_{n+1} = un(u_{n-1}^2 - 2) - u_1 for n = 1, 2, ... .
Prove that \[un\] = 2^(2^n - (-1)^n)/3, where \[x\] denotes the greatest integer
less than or equal to x.
-/

namespace Imo1976P6

theorem imo1976_p6 (u : ℕ → ℝ)
  (h₀ : u 0 = 2)
  (h₁ : u 1 = 5 / 2)
  (h₂ : ∀ n, u (n + 2) = u (n + 1) * ((u n)^2 - 2) - u 1) :
    ∀ n, ⌊u n⌋  = (2:ℝ) ^((2^n - (-1 : ℝ)^n) / 3):= sorry


end Imo1976P6
