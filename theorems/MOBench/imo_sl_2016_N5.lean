
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2016 N5

Fix some $k, a ∈ ℤ$ with $k ≥ 0$ and $a > 0$.
A pair $(x, y) ∈ ℤ^2$ is called *nice* if $(k + 1) y^2 - k x^2 = a$.
Prove that the following two statements are equivalent:
* There exists a nice pair $(x, y)$ with $x ≥ 0$ and $x^2 > a$;
* There exists a nice pair $(x, y)$ with $x ≥ 0$ and $x^2 ≤ a$.
-/
def nice (k a x y : ℤ) := (k + 1) * y ^ 2 - k * x ^ 2 = a

theorem imo_sl_2016_N5 :
    (∃ x y, 0 ≤ x ∧ a < x ^ 2 ∧ nice k a x y)
      ↔ (∃ x y, 0 ≤ x ∧ x ^ 2 ≤ a ∧ nice k a x y) := by sorry
