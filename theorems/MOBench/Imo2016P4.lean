
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2016, Problem 4

A set of positive integers is called *fragrant* if it contains
at least two elements and each of its elements has a prime
factor in common with at least one of the other elements.
Let P(n) = n² + n + 1. What is the least possible value of
positive integer b such that there exists a non-negative integer
a for which the set

  { P(a + 1), P(a + 2), ..., P(a + b) }

is fragrant?
-/
abbrev Fragrant (s : Set ℕ+) : Prop :=
  2 ≤ s.ncard ∧ ∀ m ∈ s, ∃ n ∈ s, ¬Nat.Coprime m n

abbrev P (n : ℕ) : ℕ := n^2 + n + 1

abbrev Solution : ℕ+ := 6

theorem imo2016_p4 :
    IsLeast
      {b : ℕ+ | ∃ a : ℕ, Fragrant {p | ∃ i < b, p = P (a + 1 + i)}}
      Solution := by sorry
