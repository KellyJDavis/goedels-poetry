import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the product of all positive odd integers less than $10000$?

$\text{(A)}\ \dfrac{10000!}{(5000!)^2}\qquad \text{(B)}\ \dfrac{10000!}{2^{5000}}\qquad
\text{(C)}\ \dfrac{9999!}{2^{5000}}\qquad \text{(D)}\ \dfrac{10000!}{2^{5000} \cdot 5000!}\qquad
\text{(E)}\ \dfrac{5000!}{2^{5000}}$ Show that it is \text{(D)} \dfrac{10000!}{2^{5000} \cdot 5000!}.-/
theorem amc12_2001_p5 :
    Finset.prod (Finset.filter (fun x => ¬Even x) (Finset.range 10000)) (id : ℕ → ℕ) =
      10000! / (2 ^ 5000 * 5000!) := by sorry
