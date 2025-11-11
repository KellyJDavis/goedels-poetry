import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2001, Problem 6

Let a, b, c, d be integers with a > b > c > d > 0. Suppose that

  ac + bd = (a + b - c + d) * (-a + b + c + d).

Prove that ab + cd is not prime.
-/

namespace Imo2001P6

theorem imo2001_p6 {a b c d : ℤ} (hd : 0 < d) (hdc : d < c) (hcb : c < b) (hba : b < a)
    (h : a * c + b * d = (a + b - c + d) * (-a + b + c + d)) : ¬Prime (a * b + c * d) := sorry


end Imo2001P6
