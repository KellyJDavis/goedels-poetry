
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2010 A4

Define the sequence $(x_n)_{n \ge 0}$ recursively by $x_0 = 1$,
$x_{2k} = (-1)^k x_k$, and $x_{2k + 1} = -x_k$ for all $k \in \mathbb{N}$.
Prove that for any $n \in \mathbb{N}$, $$ \sum_{i = 0}^{n-1} x_i \ge 0. $$

**Extra**: Prove that equality holds if and only if the
base $4$ representation of $n$ only contains $0$ and $2$ as its digits.
-/
/- special open -/ open Finset
/--
The sequence `x n` is defined recursively on the binary representation of `n`.
`false` corresponds to the integer value `1`, and `true` to `-1`.
-/
def x : ℕ → Bool :=
  Nat.binaryRec false fun bit k ↦ xor (bit || Nat.bodd k)

/--
The sum $S(n) = \sum_{i = 0}^{n-1} x_i$.
-/
def S (n : ℕ) : ℤ :=
  ∑ k in range n, if x k then -1 else 1

/--
This theorem states both parts of the problem:
1. The sum `S n` is always non-negative.
2. The sum is zero if and only if the base-4 digits of `n` are all either 0 or 2.
-/
theorem imo_sl_2010_A4 (n : ℕ) :
  (0 ≤ S n) ∧ (S n = 0 ↔ ∀ c ∈ Nat.digits 4 n, c = 0 ∨ c = 2) := by sorry
