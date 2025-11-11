import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- Let $S = 2010 + 2011 + \cdots + 4018$. Compute the residue of $S$, modulo 2009. Show that it is 0.-/
theorem mathd_numbertheory_353 (s : ℕ) (h₀ : s = ∑ k in Finset.Icc 2010 4018, k) : s % 2009 = 0 := by sorry
