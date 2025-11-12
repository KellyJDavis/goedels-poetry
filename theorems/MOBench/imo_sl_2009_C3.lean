
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2009 C3

Let $\{0, 1\}^*$ denote the set of finite-length binary words with letters $0$ and $1$.
Let $ε$ denote the empty word.

Define the function $f : \{0, 1\}^* → ℕ$ recursively by $f(ε) = 1$, $f(0) = f(1) = 7$, and
$$ f(wa0) = 2 f(wa) + 3 f(w) \quad \text{and} \quad f(wa1) = 3 f(wa) + f(w). $$
Fix a word $w ∈ L$, and let $w'$ denote the reversal of $w$.
Prove that $f(w') = f(w)$.
-/
/- special open -/ open List
def f : List Bool → Nat × Nat :=
  foldr (λ a (x, y) ↦ (y, match a with | false => 2 * x + 3 * y | true => 3 * x + y)) (1, 7)

theorem imo_sl_2009_C3 : ∀ l, (f l.reverse).2 = (f l).2 := by sorry
