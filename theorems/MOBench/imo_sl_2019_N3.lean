
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2019 N3

A set $S ⊆ ℤ$ is called *rootiful* if for any $a_0, a_1, … a_n ∈ S$, not all zero,
  and $x ∈ ℤ$ such that $a_0 + a_1 x + … + a_n x^n = 0$, we have $x ∈ S$.

Fix an integer $N$ with $|N| > 1$.
Find all rootiful sets containing $N^{a + 1} - N^{b + 1}$ for all $a, b ∈ ℕ$.
-/
/- special open -/ open List
def rootiful (S : Set ℤ) :=
  ∀ (x : ℤ) (P : List ℤ) (_ : ∀ a : ℤ, a ∈ P → a ∈ S) (_ : ∃ a : ℤ, a ∈ P ∧ a ≠ 0),
    P.foldr (· + x * ·) 0 = 0 → x ∈ S

theorem imo_sl_2019_N3 {N : ℤ} (h : 1 < |N|) {S : Set ℤ} :
    (rootiful S ∧ ∀ a b : ℕ, N ^ (a + 1) - N ^ (b + 1) ∈ S) ↔ S = Set.univ := by sorry
