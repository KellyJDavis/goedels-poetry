import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 2001, Problem 3

Let a,b,c ≥ 0 be real numbers satsifying

        a² + b² + c² + abc = 4.

Show that

        0 ≤ ab + bc + ca - abc ≤ 2.
-/

namespace Usa2001P3

theorem usa2001_p3 (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c)
    (h : a^2 + b^2 + c^2 + a * b * c = 4) :
    0 ≤ a * b + b * c + c * a - a * b * c ∧
    a * b + b * c + c * a - a * b * c ≤ 2 := sorry


end Usa2001P3
