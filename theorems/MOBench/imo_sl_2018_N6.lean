
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2018 N6

Let $f : \mathbb{N}^+ \to \mathbb{N}^+$ be a function such that $f(m + n)$ divides $f(m) + f(n)$
for all $m, n \in \mathbb{N}^+$.

Prove that there exists $n_0 \in \mathbb{N}^+$ such that $f(n_0)$ divides $f(n)$ for all
$n \in \mathbb{N}^+$.
-/
def IsGood (f : ℕ+ → ℕ+) : Prop :=
  ∀ m n, f (m + n) ∣ f m + f n

theorem imo_sl_2018_N6 (f : ℕ+ → ℕ+) (hf : IsGood f) :
  ∃ n₀, ∀ n, f n₀ ∣ f n := by sorry
