
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2019, Problem 1
Let ℤ be the set of integers. Determine all functions f : ℤ → ℤ such that,
for all integers a and b,￼

   f(2 * a) + 2 * f(b) = f(f(a + b)).
-/
abbrev solution_set : Set (ℤ → ℤ) :=
  { f | (∀ z, f z = 0) ∨ ∃ c, ∀ z, f z = 2 * z + c }

theorem imo2019_p1 (f : ℤ → ℤ) :
    (∀ a b, f (2 * a) + 2 * (f b) = f (f (a + b))) ↔ f ∈ solution_set := by sorry
