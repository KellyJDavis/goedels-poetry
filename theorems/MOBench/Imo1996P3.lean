
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1996, Problem 3

Let S denote the set of nonnegative integers. Find
all functions f from S to itself such that

 f(m + f(n)) = f(f(m)) + f(n)

for all m,n in S.
-/
abbrev SolutionSet : Set (ℕ → ℕ) :=
  {f : ℕ → ℕ | ∃ (k : ℕ) (n : Fin k → ℕ),
    (k = 0 ∧ f = λ _ => 0) ∨
    (k > 0 ∧ ∀ (q r : ℕ) (h : r < k), f (q * k + r) = q * k + n ⟨r, h⟩ * k)}

theorem imo1996_p3 (f : ℕ → ℕ) :
    f ∈ SolutionSet ↔ ∀ m n, f (m + f n) = f (f m) + f n := by sorry
