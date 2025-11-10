import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2004, Problem 2

Find all polynomials P with real coefficients such that
for all reals a,b,c such that ab + bc + ca = 0 we have

    P(a - b) + P(b - c) + P(c - a) = 2P(a + b + c).
-/

namespace Imo2004P2

/- determine -/ abbrev SolutionSet : Set (Polynomial ℝ) := sorry

theorem imo2004_p2 (P : Polynomial ℝ) :
    P ∈ SolutionSet ↔
    ∀ a b c, a * b + b * c + c * a = 0 →
      P.eval (a - b) + P.eval (b - c) + P.eval (c - a) =
      2 * P.eval (a + b + c) := sorry


end Imo2004P2
