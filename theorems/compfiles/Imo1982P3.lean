import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1982, Problem 3

Consider infinite sequences $\{x_n \}$ of positive reals such that $x_0 = 0$ and
$x_0 \geq x_1 \geq x_2 \geq ...$

a) Prove that for every such sequence there is an $n \geq 1$ such that:

$\frac{x_0^2}{x_1} + \ldots + \frac{x_{n-1}^2}{x_n} \geq 3.999$

b) Find such a sequence such that for all n:

$\frac{x_0^2}{x_1} + \ldots + \frac{x_{n-1}^2}{x_n} < 4$
-/

namespace Imo1982Q3

theorem imo1982_q3a {x : ℕ → ℝ} (hx : Antitone x) (h0 : x 0 = 1) (hp : ∀ k, 0 < x k) :
    ∃ n : ℕ, 3.999 ≤ ∑ k ∈ Finset.range n, (x k) ^ 2 / x (k + 1) := sorry

noncomputable /- determine -/ abbrev sol : ℕ → ℝ := sorry

theorem imo1982_q3b : Antitone sol ∧ sol 0 = 1 ∧ (∀ k, 0 < sol k)
    ∧ ∀ n, ∑ k ∈ Finset.range n, sol k ^ 2 / sol (k + 1) < 4 := sorry
