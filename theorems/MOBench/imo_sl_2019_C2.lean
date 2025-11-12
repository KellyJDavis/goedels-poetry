
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# IMO 2019 C2

Let $G$ be a totally ordered abelian group, and fix a non-negative element $g \in G$.
For a multiset $S$ of elements of $G$, let $\Sigma_S$ denote the sum of the elements of $S$,
counting multiplicity.

Let $S$ be a multiset of elements of $G$ such that $\Sigma_S \le 2|S|g$. Suppose that
each element of $S$ is greater than or equal to $g$.

Prove that for any $r \in G$ with $-2g \le r \le \Sigma_S$, there exists a sub-multiset
$S'$ of $S$ such that $r \le \Sigma_{S'} \le r + 2g$.
-/
/- special open -/ open Multiset
theorem imo_sl_2019_C2 [LinearOrderedAddCommGroup G] (g : G) (hg : 0 ≤ g) (S : Multiset G)
    (hS_elems : ∀ x ∈ S, g ≤ x)
    (hS_sum : S.sum ≤ (2 * card S) • g)
    (r : G) (hr_lower : -(2 • g) ≤ r) (hr_upper : r ≤ S.sum) :
    ∃ T ≤ S, r ≤ T.sum ∧ T.sum ≤ r + 2 • g := by sorry
