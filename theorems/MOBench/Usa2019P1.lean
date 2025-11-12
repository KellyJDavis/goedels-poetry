
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# USA Mathematical Olympiad 2019, Problem 1

Let ℕ+ be the set of positive integers.
A function f: ℕ+ → ℕ+ satisfies the equation

  fᶠ⁽ⁿ⁾(n)⬝f²(n) = n^2

for all positive integers n, where fᵏ(m) means f iterated k times on m.
Given this information, determine all possible values of f(1000).
-/
abbrev solution_set : Set ℕ+ := { x : ℕ+ | Even x.val }

theorem usa2019_p1 (m : ℕ+) :
   m ∈ solution_set ↔
    (∃ f : ℕ+ → ℕ+,
      (∀ n, f^[f n] n * f (f n) = n ^ 2) ∧
      m = f 1000) := by sorry
