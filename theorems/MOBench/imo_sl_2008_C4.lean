
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2008 C4

Let $n$ and $d$ be positive integers. Consider $2n$ lamps labelled with a pair $(b, m)$ where
$b \in \{0, 1\}$ and $m \in \{0, 1, \ldots, n - 1\}$. Initially, all the lamps are off.

Consider sequences of $k = 2d + n$ steps, where at each step, one of the lamps is switched
(off to on and vice versa).

Let $S_N$ be the set of $k$-step sequences ending in a state where the lamps labelled $(b, m)$
are on if and only if $b = 0$.

Let $S_M \subseteq S_N$ consist of the sequences that do not touch the lamps labelled
$(0, m)$ at all.

Find the ratio $|S_N|/|S_M|$.
-/
/- special open -/ open Finset
variable (I Λ : Type) [Fintype I] [Fintype Λ]

def IsNSequence [DecidableEq I] [DecidableEq Λ] (f : I → Fin 2 × Λ) : Prop :=
  ∀ p : Fin 2 × Λ, (univ.filter (f · = p)).card % 2 = p.1.val

noncomputable instance IsNSequence.instDecidablePred [DecidableEq I] [DecidableEq Λ] :
  DecidablePred (IsNSequence I Λ) := by
  unfold IsNSequence; infer_instance

def IsMSequence [DecidableEq I] [DecidableEq Λ] (f : I → Λ) : Prop :=
  ∀ l : Λ, (univ.filter (f · = l)).card % 2 = 1

noncomputable instance IsMSequence.instDecidablePred [DecidableEq I] [DecidableEq Λ] :
  DecidablePred (IsMSequence I Λ) := by
  unfold IsMSequence; infer_instance

theorem imo_sl_2008_C4 [DecidableEq I] [DecidableEq Λ] :
    Fintype.card { f : I → Fin 2 × Λ // IsNSequence I Λ f } =
      2 ^ (Fintype.card I - Fintype.card Λ) * Fintype.card { f : I → Λ // IsMSequence I Λ f } := by sorry
