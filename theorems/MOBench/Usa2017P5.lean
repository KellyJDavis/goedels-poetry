
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 2017, Problem 5

Determine the set of positive real numbers c such that there exists
a labeling of the lattice points in ℤ² with positive integers for which:

  1. only finitely many distinct labels occur, and
  2. for each label i, the distance between any two points labeled i
     is at most cⁱ.
-/
abbrev solution_set : Set ℝ := {c : ℝ | 0 < c ∧ c < Real.sqrt 2}

noncomputable def _dist : ℤ × ℤ → ℤ × ℤ → ℝ
| ⟨x1, y1⟩, ⟨x2, y2⟩ => Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

theorem usa2017_p5 (c : ℝ) :
    c ∈ solution_set ↔
    (0 < c ∧
     ∃ l : ℤ × ℤ → ℕ,
       (Set.range l).Finite ∧
       (∀ p, 0 < l p) ∧
       ∀ {p1 p2}, p1 ≠ p2 → (l p1 = l p2) →
            _dist (l p1) (l p2) ≤ c ^ (l p1)) := by sorry
