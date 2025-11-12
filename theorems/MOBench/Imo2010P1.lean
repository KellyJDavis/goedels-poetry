
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2010, Problem 1

Determine all functions f : ℝ → ℝ such that for all x,y ∈ ℝ,

               f(⌊x⌋y) = f(x)⌊f(y)⌋.
-/
abbrev solution_set : Set (ℝ → ℝ) :=
  { f | (∃ C, ⌊C⌋ = 1 ∧ f = Function.const _ C) ∨ f = Function.const _ 0 }

theorem imo2010_p1 (f : ℝ → ℝ) :
    f ∈ solution_set ↔ ∀ x y, f (⌊x⌋ * y) = f x * ⌊f y⌋ := by sorry
