
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1983, Problem 1

Let ℝ+ be the set of positive real numbers.

Determine all functions f : ℝ+ → ℝ+ which satisfy:
 i) f(xf(y)) = yf(x) for all x y ∈ ℝ+.
 ii) f(x) → 0 as x → ∞.
-/
abbrev PosReal : Type := { x : ℝ // 0 < x }

notation "ℝ+" => PosReal

abbrev SolutionSet : Set (ℝ+ → ℝ+) := { fun x ↦ 1 / x }

theorem imo1983_p1 (f : ℝ+ → ℝ+) :
    f ∈ SolutionSet ↔
    ((∀ x y, f (x * f y) = y * f x) ∧
     (∀ ε, ∃ x, ∀ y, x < y → f y < ε)) := by sorry
