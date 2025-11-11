import Mathlib.Tactic

/-!
# International Mathematical Olympiad 2022, Problem 5

Determine all possible triples of positive integers a,b,p that satisfy

  aᵖ = b! + p

where p is prime.

-/

namespace Imo2022P5

/- determine -/ abbrev solution_set : Set (ℕ × ℕ × ℕ) := sorry

theorem imo2022_p5 (a b p : ℕ) (ha : 0 < a) (hb : 0 < b) (hp : p.Prime) :
    ⟨a,b,p⟩ ∈ solution_set ↔ a^p = Nat.factorial b + p := sorry


end Imo2022P5
