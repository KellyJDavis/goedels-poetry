import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# International Mathematical Olympiad 2001, Problem 2

Let a, b, c be positive reals. Prove that

    a / √(a² + 8bc) + b / √(b² + 8ca) + c / √(c² + 8ab) ≥ 1.
-/

open Real

namespace Imo2001P2

variable {a b c : ℝ}

theorem imo2001_p2 (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 1 ≤
    a / sqrt (a ^ 2 + 8 * b * c) + b / sqrt (b ^ 2 + 8 * c * a) +
    c / sqrt (c ^ 2 + 8 * a * b) := sorry


end Imo2001P2
