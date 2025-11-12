
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2008 A3

Let $α$ be a totally ordered type.
A **Spanish couple** on $α$ is a pair of strictly increasing functions $(f, g)$
from $α$ to itself such that for all $x \in α$, $f(g(g(x))) < g(f(x))$.

Determine whether there exists a Spanish couple on:
2. The set $ℕ \times ℕ$ with the lexicographical order.
-/
structure SpanishCouple [Preorder α] (f g : α → α) : Prop where
  f_mono : StrictMono f
  g_mono : StrictMono g
  spec : f ∘ g ∘ g < g ∘ f

theorem imo_sl_2008_A3b_part2 : ∃ f g : (ℕ ×ₗ ℕ) → (ℕ ×ₗ ℕ), SpanishCouple f g := by sorry
