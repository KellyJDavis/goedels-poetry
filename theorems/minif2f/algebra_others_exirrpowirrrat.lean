import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Show that there exist real numbers $a$ and $b$ such that $a$ is irrational, $b$ is irrational, and $a^b$ is rational.-/
theorem algebra_others_exirrpowirrrat : ∃ a b, Irrational a ∧ Irrational b ∧ ¬Irrational (a ^ b) := by sorry
