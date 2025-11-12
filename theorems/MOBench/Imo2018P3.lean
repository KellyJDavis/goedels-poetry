
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2018, Problem 3

An anti-Pascal triangle is an equilateral triangular array of numbers such that,
except for the numbers in the bottom row, each number is the absolute value
of the difference of the two numbers immediately below it. For example,
the following array is an anti-Pascal triangle with four rows
which contains every integer from 1 to 10:

                  4
                2   6
              5   7   1
            8   3  10   9

Does there exist an anti-Pascal triangle with 2018 rows which contains every
integer from 1 to 1 + 2 + ... + 2018?
-/
structure Coords where
(row : ℕ) (col : ℕ)

def left_child (c : Coords) : Coords :=
 ⟨c.row.succ, c.col⟩

def right_child (c : Coords) : Coords :=
  ⟨c.row.succ, c.col.succ⟩

/--
antipascal triangle with n rows
-/
structure antipascal_triangle (n : ℕ) where
(f : Coords → ℕ)
(antipascal : ∀ x : Coords, x.row + 1 < n ∧ x.col ≤ x.row →
  f x + f (left_child x) = f (right_child x) ∨
  f x + f (right_child x) = f (left_child x))

def exists_desired_triangle : Prop :=
   ∃ t : antipascal_triangle 2018,
     ∀ n, 1 ≤ n → n ≤ ∑ i ∈ Finset.range 2018, (i + 1) →
         ∃ r, r ≤ 2018 ∧ ∃ c, c < r ∧ t.f ⟨r,c⟩ = n

abbrev does_exist : Bool := false

theorem imo2018_p3 :
    if does_exist then exists_desired_triangle else ¬ exists_desired_triangle := by sorry
