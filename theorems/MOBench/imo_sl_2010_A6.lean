
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2010 A6

Let $f, g : ℕ → ℕ$ be functions such that $f(g(x)) = f(x) + 1$
  and $g(f(x)) = g(x) + 1$ for all $x ∈ ℕ$.
Prove that $f = g$.
-/
/- special open -/ open Classical
def good (f g : ℕ → ℕ) := ∀ n : ℕ, f (g n) = (f n).succ

variable {f g : ℕ → ℕ} (h : good f g) (h0 : good g f)

theorem imo_sl_2010_A6 : f = g := by sorry
