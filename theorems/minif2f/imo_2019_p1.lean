import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- 'Let $\mathbb{Z}$ be the set of integers. Determine all functions $f : \mathbb{Z} \to \mathbb{Z}$ such that, for all
''integers $a$ and $b$, $f(2a) + 2f(b) = f(f(a + b)).$''-/
theorem imo_2019_p1 (f : ℤ → ℤ) :
    (∀ a b, f (2 * a) + 2 * f b = f (f (a + b))) ↔ ∀ z, f z = 0 ∨ ∃ c, ∀ z, f z = 2 * z + c := by sorry
