
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2017 A4

Let $G$ be a totally ordered abelian group and let $D$ be a natural number.
A sequence $(a_n)_{n \ge 0}$ of elements of $G$ satisfies the following properties:
* for any $i, j \in \mathbb{N}$ with $i + j \ge D$, we have $a_{i + j + 1} \le -a_i - a_j$,
* for any $n \ge D$, there exists $i, j \in \mathbb{N}$ such that
  $i + j = n$ and $a_{n + 1} = -a_i - a_j$.

Prove that $(a_n)_{n \ge 0}$ is bounded. Explicitly, prove that for all $n$,
$|a_n| \le 2 \max\{B, C - B\}$, where
$B = \max_{k \le D} a_k$ and $C = \max_{k \le D} (-a_k)$.
-/
variable {G : Type*} [LinearOrderedAddCommGroup G]

def seqMax (a : ℕ → G) (n : ℕ) : G :=
  if h : n = 0 then
    a 0
  else
    max (seqMax a (n-1)) (a n)

def IsGood1 (D : ℕ) (a : ℕ → G) : Prop :=
  ∀ i j : ℕ, D ≤ i + j → a (i + j + 1) ≤ -(a i) - (a j)

def IsGood2 (D : ℕ) (a : ℕ → G) : Prop :=
  ∀ n ≥ D, ∃ i j : ℕ, i + j = n ∧ a (n + 1) = -(a i) - (a j)

theorem imo_sl_2017_A4 (D : ℕ) (a : ℕ → G) (h1 : IsGood1 D a) (h2 : IsGood2 D a) (n : ℕ) :
  |a n| ≤ max ((2 : ℕ) • seqMax a D) ((2 : ℕ) • (seqMax (fun i => -a i) D - seqMax a D)) := by sorry
