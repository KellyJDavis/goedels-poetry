
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2017 N3

Let $n > 1$ be an integer.
A *special $n$-tuple* is an $n$-tuple $\mathbf{a} = (a_0, a_1, …, a_{n - 1})$ of integers
  such that there exists an indexing function $f : [n] → [n]$ such that for all $i$,
$$ n ∣ a_i + a_{i + 1} + … + a_{i + f(i)}. $$
Determine all $n > 1$ such that any special $n$-tuple $\mathbf{a}$ satisfies
$$ n ∣ a_0 + a_1 + … + a_{n-1}. $$
-/
/- special open -/ open Finset Function
structure SpecialTuple (n : ℕ) where
  toFun : Fin n.pred.succ → ℤ
  jump_shift : Fin n.pred.succ → Fin n.pred.succ
  jump_shift_spec : ∀ i, (n : ℤ) ∣ ∑ j ∈ Ico i.1 (i.1 + ((jump_shift i).1 + 1)), toFun j

def sum (X : SpecialTuple n) : ℤ := ∑ i, X.toFun i

theorem imo_sl_2017_N3 (hn : 1 < n) : (∀ X : SpecialTuple n, (n : ℤ) ∣ sum X) ↔ n.Prime := by sorry
