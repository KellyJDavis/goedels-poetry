import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic.Polyrith

/-!
# International Mathematical Olympiad 2006, Problem 3

Determine the least real number $M$ such that
$$
\left| ab(a^2 - b^2) + bc(b^2 - c^2) + ca(c^2 - a^2) \right|
≤ M (a^2 + b^2 + c^2)^2
$$
for all real numbers $a$, $b$, $c$.
-/

open Real

namespace Imo2006P3

noncomputable /- determine -/ abbrev solution : ℝ := sorry

theorem imo2006_p3 :
    IsLeast
      { M | (∀ a b c : ℝ,
              |a * b * (a ^ 2 - b ^ 2) + b * c * (b ^ 2 - c ^ 2) + c * a * (c ^ 2 - a ^ 2)| ≤
              M * (a ^ 2 + b ^ 2 + c ^ 2) ^ 2) } solution := sorry



end Imo2006P3
