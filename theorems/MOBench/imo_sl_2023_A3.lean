
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2023 A3

Let $F$ be a totally ordered field and $N$ be a positive integer.
Let $x_0, x_1, \dots, x_{N-1} \in F$ be distinct positive elements.
Suppose that for each $n \in \{0, \dots, N\}$, there exists an integer $a_n \in \mathbb{N}$ such that
$$ \left(\sum_{i=0}^{n-1} x_i\right) \left(\sum_{i=0}^{n-1} \frac{1}{x_i}\right) = a_n^2. $$
Prove that $a_N \ge \lfloor 3N/2 \rfloor$.
-/
/- special open -/ open Finset
structure GoodSeq (N : ℕ) (F : Type*) [LinearOrderedField F] where
  x : ℕ → F
  a : ℕ → ℕ
  x_pos : ∀ i < N, 0 < x i
  x_inj : ∀ i < N, ∀ j < N, x i = x j → i = j
  spec : ∀ i ≤ N, (a i : F) ^ 2 = (∑ j ∈ range i, x j) * (∑ j ∈ range i, (x j)⁻¹)

theorem imo_sl_2023_A3 {N : ℕ} {F : Type*} [LinearOrderedField F]
    (hN : 0 < N) (X : GoodSeq N F) :
  3 * N / 2 ≤ X.a N := by sorry
