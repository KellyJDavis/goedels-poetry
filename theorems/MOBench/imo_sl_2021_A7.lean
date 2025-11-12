
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2021 A7

Let $R$ be a totally ordered commutative ring.
Let $(x_n)_{n ≥ 0}$ be a sequence of elements of $R$ such that, for each $n ∈ ℕ$,
$$ x_{n + 1} x_{n + 2} ≥ x_n^2 + 1. $$
Show that for any $N ∈ ℕ$,
$$ 27 (x_0 + x_1 + … + x_{N + 1})^2 > 8 N^3. $$
-/
/- special open -/ open Finset
variable [LinearOrderedField R] [ExistsAddOfLE R]

theorem imo_sl_2021_A7 {x : ℕ → R} (hx : ∀ n, 0 ≤ x n)
    (hx0 : ∀ n, x n ^ 2 + 1 ≤ x (n + 1) * x (n + 2)) (N) :
    8 * N ^ 3 < 27 * (range (N + 2)).sum x ^ 2 := by sorry
