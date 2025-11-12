
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1968, Problem 5

Let f be a real-valued function defined for all real numbers x such that,
for some positive constant a, the equation

  f(x + a) = a/2 + √(f(x) - (f(x))²)

holds for all x.

(b) For a = 1, give an example of a non-constant function with the required properties.
-/
abbrev P (a : ℝ) (f : ℝ → ℝ) : Prop :=
  0 < a ∧
  ∀ x, (f x)^2 ≤ f x ∧ f (x + a) = 1/2 + √(f x - (f x)^2)

noncomputable abbrev solution_func : ℝ → ℝ := fun x ↦
 if Even ⌊x⌋ then 1 else 1/2

theorem imo1968_p5b :
    P 1 solution_func ∧ ¬∃c, solution_func = Function.const ℝ c := by sorry
