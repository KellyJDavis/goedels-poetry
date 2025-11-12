
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2004, Problem 2

Find all polynomials P with real coefficients such that
for all reals a,b,c such that ab + bc + ca = 0 we have

    P(a - b) + P(b - c) + P(c - a) = 2P(a + b + c).
-/
abbrev SolutionSet : Set (Polynomial ℝ) :=
  {P | ∃ (a₂ a₄ : ℝ), P = Polynomial.monomial 2 a₂ + Polynomial.monomial 4 a₄}

theorem imo2004_p2 (P : Polynomial ℝ) :
    P ∈ SolutionSet ↔
    ∀ a b c, a * b + b * c + c * a = 0 →
      P.eval (a - b) + P.eval (b - c) + P.eval (c - a) =
      2 * P.eval (a + b + c) := by sorry
