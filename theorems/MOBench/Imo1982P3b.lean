
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1982, Problem 3

Consider infinite sequences $\{x_n \}$ of positive reals such that $x_0 = 0$ and
$x_0 \geq x_1 \geq x_2 \geq ...$

b) Find such a sequence such that for all n:

$\frac{x_0^2}{x_1} + \ldots + \frac{x_{n-1}^2}{x_n} < 4$
-/
noncomputable abbrev sol : ℕ → ℝ := fun k ↦ 2⁻¹ ^ k

theorem imo1982_q3b : Antitone sol ∧ sol 0 = 1 ∧ (∀ k, 0 < sol k)
    ∧ ∀ n, ∑ k ∈ Finset.range n, sol k ^ 2 / sol (k + 1) < 4 := by sorry
