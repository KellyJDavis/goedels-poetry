
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2023, Problem 5

Let n be a positive integer. A _Japanese triangle_ is defined as
a set of 1 + 2 + ... + n dots arranged as an equilateral
triangle. Each dot is colored white or red, such that each row
has exactly one red dot.

A _ninja path_ is a sequence of n dots obtained by starting in the
top row (which has length 1), and then at each step going to one of
the dot immediately below the current dot, until the bottom
row is reached.

In terms of n, determine the greatest k such that in each Japanese triangle
there is a ninja path containing at least k red dots.
-/
structure JapaneseTriangle (n : ℕ) where
  red : (i : Finset.Icc 1 n) → Fin i.val

def next_row {n} (i : Finset.Icc 1 n) (h : i.val + 1 ≤ n) : Finset.Icc 1 n :=
  ⟨i.val + 1, by aesop⟩

structure NinjaPath (n : ℕ) where
  steps : (i : Finset.Icc 1 n) → Fin i.val
  steps_valid : ∀ i : Finset.Icc 1 n, (h : i.val + 1 ≤ n) →
     ((steps i).val = steps (next_row i h) ∨
      (steps i).val + 1 = steps (next_row i h))

noncomputable abbrev solution_value (n : ℕ) : ℕ := Nat.floor (Real.log n / Real.log 2) + 1

theorem imo2023_p5 (n : ℕ) :
    IsGreatest {k | ∀ j : JapaneseTriangle n,
                    ∃ p : NinjaPath n,
                      k ≤ Fintype.card {i // j.red i = p.steps i}}
               (solution_value n) := by sorry
