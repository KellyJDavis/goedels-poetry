
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2007 A2

A function $f : \mathbb{N}^+ \to \mathbb{N}^+$ is called *good* if, for any positive integers $m, n$, the following inequality holds:
$$ f(m + n) + 1 \ge f(m) + f(f(n)). $$
For any given $N \in \mathbb{N}^+$, determine all possible values of $k \in \mathbb{N}^+$ for which there exists a good function $f$ such that $f(N) = k$.

The solution states that:
- If $N=1$, the only possible value is $k=1$.
- If $N > 1$, the possible values are all $k$ such that $k \le N+1$.
-/
/- special open -/ open Finset
/--
A function `f : ℕ+ → ℕ+` is "good" if it satisfies the problem's inequality.
Note the inequality is rearranged to use `≤` to align with Lean's conventions.
-/
def goodPNat (f : ℕ+ → ℕ+) := ∀ m n, f m + f (f n) ≤ f (m + n) + 1

theorem imo_sl_2007_A2 {N k : ℕ+} :
  (∃ f : ℕ+ → ℕ+, goodPNat f ∧ f N = k) ↔ if N = 1 then k = 1 else k ≤ N + 1 := by sorry
