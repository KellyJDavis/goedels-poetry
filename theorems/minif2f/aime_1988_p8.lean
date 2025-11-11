import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The function $f$, defined on the set of ordered pairs of positive integers, satisfies the following properties:
$ f(x, x) = x,\; f(x, y) = f(y, x), {\rm \ and\ } (x+y)f(x, y) = yf(x, x+y). $
Calculate $f(14,52)$. Show that it is 364.-/
theorem aime_1988_p8 (f : ℕ → ℕ → ℝ) (h₀ : ∀ x, 0 < x → f x x = x)
    (h₁ : ∀ x y, 0 < x ∧ 0 < y → f x y = f y x)
    (h₂ : ∀ x y, 0 < x ∧ 0 < y → (↑x + ↑y) * f x y = y * f x (x + y)) : f 14 52 = 364 := by sorry
