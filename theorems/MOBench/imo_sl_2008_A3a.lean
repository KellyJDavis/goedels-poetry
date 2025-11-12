
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
1. The set of natural numbers $ℕ$.
-/
structure SpanishCouple [Preorder α] (f g : α → α) : Prop where
  f_mono : StrictMono f
  g_mono : StrictMono g
  spec : f ∘ g ∘ g < g ∘ f

theorem imo_sl_2008_A3a_part1 : ¬ ∃ f g : ℕ → ℕ, SpanishCouple f g := by sorry
