
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
Polish Mathematical Olympiad 2016, Stage 1, Problem 8
Author of the problem: Nguyen Hung Son
Source of the problem: https://om.sem.edu.pl/static/app_main/problems/om68_1r.pdf

Let a, b, c be integers. Show that there exists a positive integer n, such that

  n³ + an² + bn + c

is not a square of any integer.
-/
theorem poland2016_s1_p8 (a b c : ℤ) : ∃ n : ℤ, n > 0 ∧ ¬ IsSquare (n^3 + a * n^2 + b * n + c) := by sorry
