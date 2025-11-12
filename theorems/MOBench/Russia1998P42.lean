
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
 Russian Mathematical Olympiad 1998, problem 42

 A binary operation ⋆ on real numbers has the property that
 (a ⋆ b) ⋆ c = a + b + c.

 Prove that a ⋆ b = a + b.

-/
variable (star : ℝ → ℝ → ℝ)

local infixl:80 " ⋆ " => star

theorem russia1998_p42
  (stardef : ∀ a b c, a ⋆ b ⋆ c = a + b + c) :
  (∀ a b, a ⋆ b = a + b) := by sorry
