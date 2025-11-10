import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1965, Problem 1

Determine all values x in the interval 0 ≤ x ≤ 2π which
satisfy the inequality

   2 cos x ≤ |√(1 + sin 2x) − √(1 − sin 2x)| ≤ √2.
-/

namespace Imo1965P1

open Real Set

/- determine -/ abbrev the_answer : Set ℝ := sorry

theorem imo1965_p1 :
    {x ∈ Icc 0 (2*π) |
       |√(1 + sin (2*x)) - √(1 - sin (2*x))| ∈ Icc (2 * cos x) √2}
     = the_answer := sorry
