import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- The function $f(n)$ is defined on the positive integers and takes non-negative integer values. $f(2)=0,f(3)>0,f(9999)=3333$ and for all $m,n:$$ f(m+n)-f(m)-f(n)=0 \text{ or } 1. $Determine $f(1982)$. Show that it is 660.-/
theorem imo_1982_p1 (f : ℕ → ℕ)
    (h₀ : ∀ m n, 0 < m ∧ 0 < n → f (m + n) - f m - f n = 0 ∨ f (m + n) - f m - f n = 1)
    (h₁ : f 2 = 0) (h₂ : 0 < f 3) (h₃ : f 9999 = 3333) : f 1982 = 660 := by sorry
