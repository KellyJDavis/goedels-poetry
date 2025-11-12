
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2015 A1

Let $F$ be a totally ordered field.
Let $(a_n)_{n ≥ 0}$ be a sequence of positive elements of $F$ such that
  $a_{k + 1} ≥ \dfrac{(k + 1) a_k}{a_k^2 + k}$ for all $k ∈ ℕ$.
Prove that, for every $n ≥ 2$,
$$ a_0 + a_1 + … + a_{n - 1} ≥ n. $$

### Further directions

Generalize to totally ordered semirings `R` with `ExistsAddOfLE R`.

If impossible, we can alternatively generalize the above sequence to
  two sequences $(a_n)_{n ≥ 0}$, $(b_n)_{n ≥ 0}$ satisfying
  $b_{k + 1} ≤ a_k + b_k$ and $a_k b_k ≥ k$ for all $k ∈ ℕ$.
-/
/- special open -/ open Finset
theorem imo_sl_2015_A1 [LinearOrderedField F]
    {a : ℕ → F} (h : ∀ k : ℕ, 0 < a k)
    (h0 : ∀ k : ℕ, ((k.succ : F) * a k) / (a k ^ 2 + (k : F)) ≤ a k.succ) :
    ∀ n : ℕ, 2 ≤ n → (n : F) ≤ (range n).sum (fun i => a i) := by sorry
