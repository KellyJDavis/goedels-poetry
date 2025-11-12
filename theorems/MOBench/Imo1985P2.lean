
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1985, Problem 5
Each of the numbers in the set $N=\{1, 2, 3, \dots, n-1\}$,
where $n ≥ 3$, is colored with one of two colors, say red or black,
so that:

  1. $i$ and $n-i$ always receive the same color, and
  2. for some $j ∈ N$ relatively prime to $n$, $i$ and $|j-i|$ receive
     the same color

for all $i ∈ N, i ≠ j$.

Prove that all numbers in $N$ must receive the same color.
-/
/-- The conditions on the problem's coloring `C`.
Although its domain is all of `ℕ`, we only care about its values in `Set.Ico 1 n`. -/
def Condition (n j : ℕ) (C : ℕ → Fin 2) : Prop :=
  (∀ i ∈ Set.Ico 1 n, C i = C (n - i)) ∧
  ∀ i ∈ Set.Ico 1 n, i ≠ j → C i = C (j - i : ℤ).natAbs

theorem imo2001_p3 {n j : ℕ} (hn : 3 ≤ n) (hj : j ∈ Set.Ico 1 n)
    (cpj : Nat.Coprime n j) {C : ℕ → Fin 2} (hC : Condition n j C)
    {i : ℕ} (hi : i ∈ Set.Ico 1 n) :
    C i = C j := by sorry
