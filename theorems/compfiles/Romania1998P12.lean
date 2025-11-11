import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Tactic

/-!
# Romanian Mathematical Olympiad 1998, Problem 12

Find all functions u : ℝ → ℝ for which there exists a strictly monotonic
function f : ℝ → ℝ such that

  ∀ x,y ∈ ℝ, f(x + y) = f(x)u(y) + f(y)
-/

namespace Romania1998P12

/- determine -/ abbrev solution_set : Set (ℝ → ℝ) := sorry

theorem romania1998_p12 (u : ℝ → ℝ) :
    (∃ f : ℝ → ℝ, (StrictMono f ∨ StrictAnti f)
          ∧ ∀ x y : ℝ, f (x + y) = f x * u y + f y) ↔
    u ∈ solution_set := sorry


end Romania1998P12
