
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2013 A5

Find all functions $f : \mathbb{N} \to \mathbb{N}$ such that for any $n \in \mathbb{N}$,
$$ f(f(f(n))) = f(n + 1) + 1. $$
-/
def IsGood (f : ℕ → ℕ) : Prop :=
  ∀ n, f^[3] n = f (n + 1) + 1

def answer2 : ℕ → ℕ
  | 0 => 1
  | 1 => 6
  | 2 => 3
  | 3 => 0
  | n + 4 => answer2 n + 4

theorem imo_sl_2013_A5 (f : ℕ → ℕ) :
  IsGood f ↔ f = Nat.succ ∨ f = answer2 := by sorry
