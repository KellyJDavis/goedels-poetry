import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2000, Problem 2

Let a, b, c be positive real numbers such that abc = 1. Show that

    (a - 1 + 1/b)(b - 1 + 1/c)(c - 1 + 1/a) ≤ 1.
-/

namespace Imo2000P2

theorem imo2000_p2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (habc : a * b * c = 1) :
    (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := sorry


end Imo2000P2
