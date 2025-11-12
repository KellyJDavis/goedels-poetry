
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1960, Problem 2

For what values of the variable $x$ does the following inequality hold:

\[\dfrac{4x^2}{(1 - \sqrt {2x + 1})^2} < 2x + 9 \ ?\]
-/
/- special open -/ open Set






/-- The predicate says that `x` satisfies the inequality

\[\dfrac{4x^2}{(1 - \sqrt {2x + 1})^2} < 2x + 9\]

and belongs to the domain of the function on the left-hand side.
-/
@[mk_iff isGood_iff']
structure IsGood (x : ℝ) : Prop where
  /-- The number satisfies the inequality. -/
  ineq : 4 * x ^ 2 / (1 - sqrt (2 * x + 1)) ^ 2 < 2 * x + 9
  /-- The number belongs to the domain of \(\sqrt {2x + 1}\). -/
  sqrt_dom : 0 ≤ 2 * x + 1
  /-- The number belongs to the domain of the denominator. -/
  denom_dom : (1 - sqrt (2 * x + 1)) ^ 2 ≠ 0

abbrev SolutionSet : Set ℝ := Ico (-1/2) (45/8) \ {0}

theorem imo1960_p2 {x} : IsGood x ↔ x ∈ SolutionSet := by sorry
