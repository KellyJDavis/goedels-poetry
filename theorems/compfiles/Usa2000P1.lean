import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
USA Mathematical Olympiad 2000, Problem 1

A function f : ℝ → ℝ is called "very convex" if it satisfies

  ∀ x y : ℝ, (f(x) + f(y))/2 ≥  f((x + y)/2) + |x - y|.

Show that there exist no very convex functions.
-/

namespace Usa2000P1

theorem usa2000_p1 :
    ¬∃ f : ℝ → ℝ,
      ∀ x y : ℝ, f ((x + y) / 2) + |x - y| ≤ (f x + f y) / 2 := sorry


end Usa2000P1
