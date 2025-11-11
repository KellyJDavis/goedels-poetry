import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Sally, Wei-Hwa, and Zoe are playing a game of marbles involving first arranging as many piles of 10 marbles as possible.  Sally brought 239 marbles, Wei-Hwa brought 174 marbles, and Zoe brought 83 marbles.  If all their marbles are grouped together, how many must be removed in order to start the game? Show that it is 6.-/
theorem mathd_numbertheory_254 : (239 + 174 + 83) % 10 = 6 := by sorry
