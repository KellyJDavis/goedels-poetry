
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2011 A4

Find all functions $f, g : ℕ → ℕ$ such that, for any $k ∈ ℕ$,
$$ f^{g(k) + 2}(k) + g^{f(k) + 1}(k) + g(k + 1) + 1 = f(k + 1). $$

### Extra Notes

The original version using signature $ℕ^+ → ℕ^+$ is:
$$ f^{g(k) + 1}(k) + g^{f(k)}(k) + g(k + 1) = f(k + 1) + 1. $$
-/
/- special open -/ open Function
theorem imo_sl_2011_A4 {f g : ℕ+ → ℕ+} :
    (∀ n, f^[g n + 1] n + (g^[f n] n + g (n + 1)) = f (n + 1) + 1)
      ↔ f = id ∧ g = λ _ ↦ 1 := by sorry
