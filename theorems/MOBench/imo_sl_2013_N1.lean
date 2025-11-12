
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2013 N1

Find all functions $f : ℕ^+ → ℕ^+$ such that, for any $m, n : ℕ^+$,
$$ m^2 + f(n) ∣ m f(m) + n. $$
-/
theorem imo_sl_2013_N1 {f : ℕ+ → ℕ+} :
    (∀ m n : ℕ+, m * m + f n ∣ m * f m + n) ↔ f = id := by sorry
