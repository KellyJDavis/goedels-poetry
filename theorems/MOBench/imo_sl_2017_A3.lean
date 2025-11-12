
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2017 A3

Let $S$ be a finite set, and fix some $f : S → S$.
Suppose that, for any $g : S → S$, $$f ∘ g ∘ f = g ∘ f ∘ g \implies g = f. $$
Prove that $f^2(S) = f(S)$.
-/
theorem imo_sl_2017_A3 (h : ∀ g : S → S, f ∘ g ∘ f = g ∘ f ∘ g → g = f) :
    Set.range f^[2] = Set.range f := by sorry
