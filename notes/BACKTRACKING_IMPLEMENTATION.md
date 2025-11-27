# Proof Backtracking Implementation

## Overview

This document describes the implementation of the proof backtracking feature for Gödel's Poetry. The backtracking mechanism allows the system to recover from failed proof attempts by re-sketching ancestor nodes with different decomposition strategies.

## Problem Statement

Previously, when a `DecomposedFormalTheoremState` reached `decomposition_attempts >= DECOMPOSER_AGENT_MAX_SELF_CORRECTIONS`, the system would simply set `is_finished = True` and give up. This was suboptimal because an ancestor of the failed node might still have remaining attempts and could be sketched in a different manner to explore alternative proof strategies.

## Solution Design

### Key Requirements

1. **Backtrack to nearest ancestor**: Find the closest ancestor with `decomposition_attempts < DECOMPOSER_AGENT_MAX_SELF_CORRECTIONS`
2. **Increment attempts on re-sketch**: When re-sketching, increment the ancestor's `decomposition_attempts`
3. **Exhaust all options**: Only set `is_finished = True` when all ancestors (including root) have exhausted their attempts
4. **Re-sketch root if needed**: The root node can be re-sketched if it's a `DecomposedFormalTheoremState`
5. **Preserve history**: Keep `decomposition_history` but use a different prompt for backtracking
6. **Immediate backtracking**: Backtrack immediately when the first child fails

### Implementation Components

#### 1. New Queue: `decomposition_backtrack_queue`

Added to `GoedelsPoetryState.__init__()`:
```python
self.decomposition_backtrack_queue: list[DecomposedFormalTheoremState] = []
```

This queue holds nodes that need to be re-sketched due to failed children attempts.

#### 2. Helper Methods in `GoedelsPoetryStateManager`

##### `_find_backtrackable_ancestor(node)`
Traverses up the tree from a failed node to find the nearest ancestor with `decomposition_attempts < DECOMPOSER_AGENT_MAX_SELF_CORRECTIONS`. Returns `None` if no such ancestor exists.

##### `_collect_all_descendants(node)`
Recursively collects all descendants (children, grandchildren, etc.) of a given node.

##### `_remove_nodes_from_all_queues(nodes)`
Removes the specified nodes from all proof and decomposition queues. This ensures that descendants of a backtracked node are completely removed from the system.

##### `_prepare_node_for_resketching(node)`
Clears a node's children, sketch, AST, and errors while preserving its `decomposition_history` and `decomposition_attempts`.

##### `_handle_failed_sketch(failed_sketch)`
Main backtracking logic:
1. Find a backtrackable ancestor
2. If none exists, set `is_finished = True`
3. Otherwise, collect all descendants, remove them from queues, prepare the ancestor for re-sketching, and queue it for backtracking

#### 3. State Manager Methods

##### `get_sketches_to_backtrack()`
Returns `DecomposedFormalTheoremStates` containing nodes that need backtrack re-sketching.

##### `set_backtracked_sketches(backtracked_sketches)`
Moves backtracked sketches from the backtrack queue to the search query generation queue. This allows search queries to be regenerated based on the failure context before re-sketching.

#### 4. Modified `set_validated_sketches()`

Changed from:
```python
# Set is_finished appropriately
self._state.is_finished = len(sketches_too_difficult) > 0
```

To:
```python
# Handle sketches that are too difficult - try backtracking
for sketch_too_difficult in sketches_too_difficult:
    self._handle_failed_sketch(sketch_too_difficult)
```

#### 5. New Supervisor Action

Added to `supervisor_agent.get_action()`:
```python
if self._state_manager.get_sketches_to_backtrack()["inputs"]:
    return "request_proof_sketches_backtrack"
```

The backtrack action is checked after corrections but before parsing, ensuring proper priority.

#### 6. New Framework Method

Added `request_proof_sketches_backtrack()` to `GoedelsPoetryFramework`:
```python
def request_proof_sketches_backtrack(self) -> None:
    """
    Requests re-sketching for proof sketches whose children failed to prove.
    """
    backtrack_agent = SketchBacktrackAgentFactory.create_agent()
    decomposed_states = self._state_manager.get_sketches_to_backtrack()
    decomposed_states = cast(DecomposedFormalTheoremStates, backtrack_agent.invoke(decomposed_states))
    self._state_manager.set_backtracked_sketches(decomposed_states)
```

#### 7. New Agent: `SketchBacktrackAgent`

Created `sketch_backtrack_agent.py` with a dedicated agent for handling backtrack re-sketching. This agent uses a specialized prompt that differs from both initial and corrective sketching.

#### 8. New Prompt: `decomposer-backtrack.md`

Created a specialized prompt that:
- Explains that the previous decomposition couldn't be proven
- Requests a completely different decomposition strategy
- Asks for analysis of why the previous approach might have failed
- Encourages exploration of alternative proof techniques

## Workflow Example

### Before Backtracking
```
Root (attempts: 0)
  └─ Node A (attempts: 5, MAX_RETRIES reached)
       └─ Child 1 (can't prove)
       └─ Child 2 (can't prove)

System: is_finished = True
```

### After Backtracking
```
Root (attempts: 0)
  └─ Node A (removed from tree)
       └─ Child 1 (removed from queues)
       └─ Child 2 (removed from queues)

Root is prepared for re-sketching and queued for backtracking:
- children cleared
- sketch, ast, errors cleared
- decomposition_history preserved
- Root is queued in decomposition_backtrack_queue

System: request_proof_sketches_backtrack() is called
Root receives new prompt asking for different strategy
Root is re-sketched with attempts incremented to 1
New decomposition is attempted...
```

## Key Benefits

1. **Resilient to dead ends**: The system can recover from unproductive decomposition strategies
2. **Hierarchical exploration**: Explores the proof search space more thoroughly by trying different approaches at different levels
3. **Preserves successful work**: Only backtracks as far as necessary, preserving valid portions of the tree
4. **Graceful degradation**: Only gives up when all possible backtracking options are exhausted

## Testing Considerations

When testing the backtracking feature:

1. **Simple case**: A theorem with one failed decomposition that can be re-sketched successfully
2. **Multi-level backtracking**: A deep tree where backtracking happens at different levels
3. **Root backtracking**: Cases where the root itself needs to be re-sketched
4. **Exhaustion**: Cases where all ancestors have exhausted their attempts (should set `is_finished = True`)
5. **Queue cleanup**: Verify that descendants are properly removed from all queues

## Future Enhancements

1. **Backtracking metrics**: Track how often backtracking occurs and at what depths
2. **Adaptive strategies**: Learn which decomposition strategies work better for certain theorem types
3. **Partial preservation**: Consider preserving some successful children when backtracking
4. **Backtracking limits**: Add a maximum backtracking depth or total backtrack count to prevent infinite loops

## Files Modified

1. `goedels_poetry/state.py` - Added queues, helper methods, and backtracking logic
2. `goedels_poetry/agents/supervisor_agent.py` - Added backtrack action
3. `goedels_poetry/framework.py` - Added backtrack method and import
4. `goedels_poetry/agents/sketch_backtrack_agent.py` - New agent (created)
5. `goedels_poetry/data/prompts/decomposer-backtrack.md` - New prompt (created)

## Summary

The backtracking implementation provides a robust mechanism for recovering from failed proof attempts by exploring alternative decomposition strategies at appropriate levels in the proof tree. The implementation follows the existing patterns in the codebase and integrates seamlessly with the multi-agent workflow.
