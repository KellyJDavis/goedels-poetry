
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# Romanian Mathematical Olympiad 1998, Problem 12

Find all functions u : ℝ → ℝ for which there exists a strictly monotonic
function f : ℝ → ℝ such that

  ∀ x,y ∈ ℝ, f(x + y) = f(x)u(y) + f(y)
-/
abbrev solution_set : Set (ℝ → ℝ) :=
  { u | ∃ k : ℝ, ∀ x : ℝ, u x = Real.exp (k * x) }

theorem romania1998_p12 (u : ℝ → ℝ) :
    (∃ f : ℝ → ℝ, (StrictMono f ∨ StrictAnti f)
          ∧ ∀ x y : ℝ, f (x + y) = f x * u y + f y) ↔
    u ∈ solution_set := by sorry
