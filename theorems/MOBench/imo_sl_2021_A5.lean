
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
### IMO 2021 A5

Let $F$ be a totally ordered field.
Let $a_1, a_2, …, a_n ∈ F$ be non-negative elements.
Let $r ∈ F$ be any positive element such that $r ≥ a_1 + a_2 + … + a_n$.
Prove that
$$ \sum_{k = 1}^n \frac{a_k}{r - a_k} (a_1 + a_2 + … + a_{k - 1})^2 < \frac{r^2}{3}. $$
-/
def targetSumPair [Field F] (r : F) (l : List F) : F × F :=
  l.foldr (λ a p ↦ (a / (r - a) * p.2 ^ 2 + p.1, a + p.2)) (0, 0)

theorem imo_sl_2021_A5 [LinearOrderedField F]
    {r : F} (hr : 0 < r) (l : List F)
    (hl : ∀ x ∈ l, 0 ≤ x) (h : l.foldr (· + ·) 0 ≤ r) :
    (targetSumPair r l).1 < r ^ 2 / 3 := by sorry
