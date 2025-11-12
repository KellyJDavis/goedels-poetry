
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 2022, Problem 5

Determine all possible triples of positive integers a,b,p that satisfy

  aᵖ = b! + p

where p is prime.

-/
abbrev solution_set : Set (ℕ × ℕ × ℕ) := { ⟨2,2,2⟩, ⟨3,4,3⟩ }

theorem imo2022_p5 (a b p : ℕ) (ha : 0 < a) (hb : 0 < b) (hp : p.Prime) :
    ⟨a,b,p⟩ ∈ solution_set ↔ a^p = Nat.factorial b + p := by sorry
