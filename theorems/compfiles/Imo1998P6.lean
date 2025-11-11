import Mathlib.Tactic

/-!
# International Mathematical Olympiad 1998, Problem 6

Consider all functions f from the set of all positive integers into itself satisfying
f(t^2f(s)) = sf(t)^2 for all s and t.
Determine the least possible value of f(1998).
-/

namespace Imo1998P6

/- determine -/ abbrev solution : ℕ+ := sorry

theorem imo1998_p6
    (f : ℕ+ → ℕ+)
    (h : ∀ s t, f (t^2 * f s) = s * (f t)^2) :
    IsLeast {n : ℕ | n = f 1998} solution := sorry


end Imo1998P6
