import Mathlib.Tactic

/-!
# USA Mathematical Olympiad 1982, Problem 4

Prove that there exists a positive integer k such that
k⬝2ⁿ + 1 is composite for every integer n.
-/

namespace Usa1982P4

theorem usa1982_p4 :
    ∃ k : ℕ, 0 < k ∧ ∀ n : ℕ, ¬ Prime (k * (2 ^ n) + 1) := sorry


end Usa1982P4
