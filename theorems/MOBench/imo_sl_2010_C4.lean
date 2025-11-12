
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2010 C4 (P5)

In the board, $N = 6$ stacks of coins are given, with each stack initially containing one coin.
At any time, one of the following operations can be performed:
* **Type 1:** Remove one coin from a non-empty stack $k+1$ and add two coins to stack $k$ (for $k < 5$).
* **Type 2:** Remove one coin from a non-empty stack $k+2$ and swap the contents of stacks $k$ and $k+1$ (for $k < 4$).

Is it possible that, after some operations, we are left with stack 0
  containing $A = 2010^{2010^{2010}}$ coins and all other stacks empty?
-/
/- special open -/ open List
inductive isReachable : List Nat → List Nat → Prop
  | type1_move (k m) : isReachable [k + 1, m] [k, m + 2]
  | type2_move (k m n) : isReachable [k + 1, m, n] [k, n, m]
  | refl (l) : isReachable l l
  | trans (h : isReachable l₁ l₂) (h : isReachable l₂ l₃) : isReachable l₁ l₃
  | append_right (h : isReachable l₁ l₂) (l) : isReachable (l₁ ++ l) (l₂ ++ l)
  | cons_left (h : isReachable l₁ l₂) (k) : isReachable (k :: l₁) (k :: l₂)

theorem imo_sl_2010_C4 :
    isReachable (replicate 6 1) (replicate 5 0 ++ [2010 ^ 2010 ^ 2010]) := by sorry
