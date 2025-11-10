import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1982, Problem 4

Prove that if n is a positive integer such that the equation

x3 - 3xy^2 + y^3 = n

has a solution in integers x, y, then it has at least three such solutions.
Show that the equation has no solutions in integers for n = 2891.
-/

namespace Imo1982P4

theorem imo1982_p4 (n : ℕ)
  (hn : 0 < n)
  (hxy : ∃ x y : ℤ, x^3 - 3 * x * y^2 + y^3 = n) :
    (n ≠ 2891) ∧
    ∃ x1 x2 x3 y1 y2 y3 : ℤ, (x1^3 - 3 * x1 * y1^2 + y1^3 = n ∧
      x2^3 - 3 * x2 * y2^2 + y2^3 = n ∧
      x3^3 - 3 * x3 * y3^2 + y3^3 = n ∧
      (x1 ≠ x2 ∨ y1 ≠ y2) ∧
      (x1 ≠ x3 ∨ y1 ≠ y3) ∧
      (x2 ≠ x3 ∨ y2 ≠ y3)) := sorry


end Imo1982P4
