
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1966, Problem 4

Prove that for every natural number n and for every real
number x that is not of the form kπ/2ᵗ for t a non-negative
integer and k any integer,

 1 / (sin 2x) + 1 / (sin 4x) + ... + 1 / (sin 2ⁿx) = cot x - cot 2ⁿ x.
-/
theorem imo1966_p4 (n : ℕ) (x : ℝ)
    (hx : ∀ t : ℕ, ∀ k : ℤ, x ≠ k * Real.pi / 2^t) :
    ∑ i ∈ Finset.range n, 1 / Real.sin (2^(i + 1) * x) =
    Real.cot x - Real.cot (2^n * x) := by sorry
