
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-!
# International Mathematical Olympiad 1964, Problem 4

Seventeen people correspond by mail with one another -- each one with
all the rest. In their letters only three different topics are
discussed. Each pair of correspondents deals with only one of the topics.
Prove that there are at least three people who write to each other
about the same topic.

-/
theorem imo1964_p4
    (Person Topic : Type)
    [Fintype Person] [DecidableEq Person]
    [Fintype Topic] [DecidableEq Topic]
    (card_person : Fintype.card Person = 17)
    (card_topic : Fintype.card Topic = 3)
    (discusses : Person → Person → Topic)
    (discussion_sym : ∀ p1 p2 : Person, discusses p1 p2 = discusses p2 p1) :
    ∃ t : Topic, ∃ s : Finset Person,
      2 < s.card ∧
        ∀ p1 ∈ s, ∀ p2 ∈ s, p1 ≠ p2 → discusses p1 p2 = t := by sorry
