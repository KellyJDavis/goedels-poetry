import Mathlib.Tactic
import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# USA Mathematical Olympiad 1974, Problem 2

Prove that if a, b, and c are positive real numbers, then
a^a * b^b * c^c ≥ (abc)^((a+b+c)/3)
-/

namespace Usa1974P2

theorem usa1974_p2 :
    ∀ (a b c : ℝ), a > 0 → b > 0 → c > 0 → a^a * b^b * c^c ≥ (a*b*c)^((a+b+c)/3) := sorry


end Usa1974P2
