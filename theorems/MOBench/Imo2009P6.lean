
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2009, Problem 6

Let a₁, a₂, ..., aₙ be distinct positive integers and let M
be a set of n - 1 positive integers not containing
s = a₁ + a₂ + ... + aₙ. A grasshopper is to jump along the
real axis, starting at the point 0 and making n jumps to
the right with lengths a₁, a₂, ..., aₙ in some order. Prove
that the order can be chosen in such a way that the
grasshopper never lands on any point in M.
-/
theorem imo2009_p6 (n : ℕ) (hn : 0 < n)
    (a : Fin n → ℤ)
    (ainj : a.Injective)
    (apos : ∀ i, 0 < a i)
    (M : Finset ℤ)
    (Mpos : ∀ m ∈ M, 0 < m)
    (Mcard : M.card = n - 1)
    (hM : ∑ i, a i ∉ M)
    : ∃ p : Equiv.Perm (Fin n),
        ∀ i : Fin n,
          ∑ j ∈ Finset.univ.filter (· ≤ i), a (p j) ∉ M := by sorry
