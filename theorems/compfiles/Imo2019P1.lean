import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2019, Problem 1
Let ℤ be the set of integers. Determine all functions f : ℤ → ℤ such that,
for all integers a and b,￼

   f(2 * a) + 2 * f(b) = f(f(a + b)).
-/

namespace Imo2019P1

/- determine -/ abbrev solution_set : Set (ℤ → ℤ) := sorry

theorem imo2019_p1 (f : ℤ → ℤ) :
    (∀ a b, f (2 * a) + 2 * (f b) = f (f (a + b))) ↔ f ∈ solution_set := sorry


end Imo2019P1
