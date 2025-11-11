import Mathlib.Algebra.Periodic
import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1968, Problem 5

Let f be a real-valued function defined for all real numbers x such that,
for some positive constant a, the equation

  f(x + a) = a/2 + √(f(x) - (f(x))²)

holds for all x.

(a) Prove that the function f is periodic.
(b) For a = 1, give an example of a non-constant function with the required properties.
-/

namespace Imo1968P5

abbrev P (a : ℝ) (f : ℝ → ℝ) : Prop :=
  0 < a ∧
  ∀ x, (f x)^2 ≤ f x ∧ f (x + a) = 1/2 + √(f x - (f x)^2)

theorem imo1968_p5a (f : ℝ → ℝ) (a : ℝ) (hf : P a f) :
    ∃ b, 0 < b ∧ f.Periodic b := sorry

noncomputable /- determine -/ abbrev solution_func : ℝ → ℝ := sorry

theorem imo1968_p5b :
    P 1 solution_func ∧ ¬∃c, solution_func = Function.const ℝ c := sorry


end Imo1968P5
