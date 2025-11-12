
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1991, Problem 6

An infinite sequence x₀,x₁,x₂,... of real numbers is said to be *bounded*
if there is a constant C such that |xᵢ| ≤ C for every i ≥ 0.

Given any real number a > 1, construct a bounded infinite sequence
x₀,x₁,x₂,... such that

                  |xᵢ - xⱼ|⬝|i - j| ≥ 1

for every pair of distinct nonnegative integers i, j.
-/
abbrev Bounded (x : ℕ → ℝ) : Prop := ∃ C, ∀ i, |x i| ≤ C

noncomputable abbrev solution (a : ℝ) (ha : 1 < a) : ℕ → ℝ :=
  let t := 1/(2^a)
  let c := 1 - t/(1 - t)
  λ n => if n = 0 then 0 else
    (1/c) * (∑ i ∈ Finset.filter (λ i => (n / 2^i) % 2 = 1) (Finset.range (Nat.log2 n + 1)), t^i)

theorem imo1991_p6 (a : ℝ) (ha : 1 < a) :
    Bounded (solution a ha) ∧
    ∀ i j, i ≠ j →
      1 ≤ |solution a ha i - solution a ha j| * |(i:ℝ) - j| := by sorry
