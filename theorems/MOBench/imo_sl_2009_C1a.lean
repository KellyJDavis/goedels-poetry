
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2009 C1

Fix non-negative integers $M$ and $n$.
Two players, $A$ and $B$, play the following game on the board.
The board consists of $M$ cards in a row, one side labelled $0$ and another side labelled $1$.

Initially, all cards are labelled $1$.
Then $A$ and $B$ take turns performing a move of the following form.
Choose an index $i ∈ ℕ$ such that $i + n < M$ and the $(i + n)^{\text{th}}$ card shows $1$.
Then flip the $j^{\text{th}}$ card for any $i ≤ j ≤ i + n$.
The last player who can make a legal move wins.

Assume that $A$ and $B$ uses the best strategy.
1. Show that the game always ends.
-/
/- special open -/ open Relation Finset
structure GameState (n : ℕ) where
  board : Finset ℕ
  numMoves : ℕ

namespace GameState

def init (M n : ℕ) : GameState n where
  board := range M
  numMoves := 0

inductive ValidMove (X : GameState n) : GameState n → Prop
  | flip (i : ℕ) (h : i + n ∈ X.board) :
      ValidMove X ⟨symmDiff X.board (Icc i (i + n)), X.numMoves.succ⟩

def IsReachable : GameState n → GameState n → Prop := ReflTransGen ValidMove

def Ends (X : GameState n) := ∀ Y : GameState n, ¬X.ValidMove Y

def P1Wins {X : GameState n} (_ : X.Ends) : Prop := X.numMoves % 2 = 1

theorem imo_sl_2009_C1a_part1 {M n : ℕ} {X : GameState n} (h : (init M n).IsReachable X) :
    X.numMoves < 2 ^ M := by sorry
