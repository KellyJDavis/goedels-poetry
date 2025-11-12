
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2007, Problem 1

Real numbers a₁, a₂, ..., aₙ are fixed. For each 1 ≤ i ≤ n,
we let dᵢ = max {aⱼ : 1 ≤ j ≤ i} - min {aⱼ : i ≤ j ≤ n},
and let d = max {dᵢ : 1 ≤ i ≤ n}.

(b) Show that there exists some choice of x₁ ≤ ... ≤ xₙ which achieves equality.
-/
noncomputable abbrev d {n : ℕ} (a : Fin n → ℝ) (i : Fin n) :=
  (⨆ j : {j // j ≤ i}, a j - ⨅ j : {j // i ≤ j}, a j)

theorem imo2007_p1b {n : ℕ} (hn : 0 < n) {a : Fin n → ℝ} :
    ∃ x : Fin n → ℝ, Monotone x ∧
      (⨆ i, d a i) / 2 = ⨆ i, |x i - a i| := by sorry
