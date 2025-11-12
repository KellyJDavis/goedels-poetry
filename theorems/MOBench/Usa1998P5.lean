
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 1998, Problem 5

Prove that for each n ≥ 2, there is a set S of n integers such that
(a-b)² divides ab for every distinct a,b ∈ S.
-/
theorem usa1998_p5 (n : ℕ) (_hn : 2 ≤ n) :
    ∃ S : Finset ℤ,
       S.card = n ∧
       ∀ a ∈ S, ∀ b ∈ S, a ≠ b → (a - b)^2 ∣ a * b := by sorry
