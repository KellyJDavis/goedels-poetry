
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1972, Problem 5

`f` and `g` are real-valued functions defined on the real line. For all `x` and `y`,
`f(x + y) + f(x - y) = 2f(x)g(y)`. `f` is not identically zero and `|f(x)| ≤ 1` for all `x`.
Prove that `|g(x)| ≤ 1` for all `x`.
-/
theorem imo1972_p5 (f g : ℝ → ℝ) (hf1 : ∀ x, ∀ y, f (x + y) + f (x - y) = 2 * f x * g y)
    (hf2 : BddAbove (Set.range fun x => ‖f x‖)) (hf3 : ∃ x, f x ≠ 0) (y : ℝ) : ‖g y‖ ≤ 1 := by sorry
