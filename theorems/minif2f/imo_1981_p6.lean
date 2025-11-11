import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The function $f(x,y)$ satisfies

(1) $f(0,y)=y+1, $

(2) $f(x+1,0)=f(x,1), $

(3) $f(x+1,y+1)=f(x,f(x+1,y)), $

for all non-negative integers $x,y $. Determine $f(4,1981) $.-/
theorem imo_1981_p6 (f : ℕ → ℕ → ℕ) (h₀ : ∀ y, f 0 y = y + 1) (h₁ : ∀ x, f (x + 1) 0 = f x 1)
    (h₂ : ∀ x y, f (x + 1) (y + 1) = f x (f (x + 1) y)) : ∀ y, f 4 (y + 1) = 2 ^ (f 4 y + 3) - 3 := by sorry
