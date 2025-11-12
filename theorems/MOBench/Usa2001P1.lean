
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 2001, Problem 1

Each of eight boxes contains six balls.
Each ball has been colored with one of n colors, such that no two balls
in the same box are the same color, and no two colors occur together in
more than one box. Determine, with justification, the smallest integer n
for which this is possible.
-/
def possible_num_colors : Set ℕ :=
{ n : ℕ | ∃ f : Fin 8 → Finset (Fin n),
    (∀ i, (f i).card = 6) ∧
    (∀ x y : Fin n, ∀ i j : Fin 8,
      i ≠ j → x ∈ f i → y ∈ f i →
        (¬ (x ∈ f j ∧ y ∈ f j))) }

abbrev min_colors : ℕ := 23

theorem usa2001_p1 : IsLeast possible_num_colors min_colors := by sorry
