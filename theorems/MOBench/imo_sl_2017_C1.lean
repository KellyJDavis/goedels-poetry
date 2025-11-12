
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2017 C1

A rectangle in $ℕ^2$ is a subset of form $\{a, a + 1, …, a + w - 1\}
  × \{b, b + 1, …, b + h - 1\}$ for some $a, b, w, h ∈ ℕ$.
Given such rectangle, the quantity $w$ and $h$ are called the
  *width* and *height* of the rectangle, respectively.

A rectangle $R$ in $ℕ^2$ with odd width and height is
  partitioned into small rectangles.
Prove that there exists a small rectangle $R'$ with the following property:
  the distances from the sides of $R'$ to the respective sides
    of $R$ all have the same parity.
-/
/- special open -/ open Finset
def latticeRect (q : (ℕ × ℕ) × ℕ × ℕ) : Finset (ℕ × ℕ) :=
  Ico q.1.1 (q.1.1 + q.2.1) ×ˢ Ico q.1.2 (q.1.2 + q.2.2)

theorem imo_sl_2017_C1 {I : Finset ι}
    (h : (I : Set ι).PairwiseDisjoint (latticeRect ∘ Q))
    (h0 : m.bodd = true ∧ n.bodd = true)
    (h1 : latticeRect ⟨⟨0, 0⟩, ⟨m, n⟩⟩ = I.disjiUnion (latticeRect ∘ Q) h) :
    ∃ i ∈ I, ((Q i).2.1.bodd = true ∧ (Q i).2.2.bodd = true)
      ∧ ((Q i).1.1 + (Q i).1.2).bodd = false := by sorry
