# Integration Notes for AST Sorries Enhancement

## Current State

The AST class now supports an optional `sorries` parameter that enables extraction of type information when generating subgoal code. This is **backward compatible** - existing code continues to work without modification.

## Agent Integration Opportunities

### Current AST Instantiation Points

1. **`sketch_parser_agent.py:125`**
   ```python
   state["ast"] = AST(ast_without_imports)
   ```

2. **`proof_parser_agent.py:123`**
   ```python
   state["ast"] = AST(ast_without_imports)
   ```

Both agents currently create AST without sorries, so they won't benefit from type information extraction yet.

### Where Sorries Are Available

1. **`sketch_checker_agent.py:118`**
   ```python
   parsed_response = parse_kimina_check_response(check_response)
   # parsed_response["sorries"] is available here
   ```

2. **`proof_checker_agent.py:116`**
   ```python
   parsed_response = parse_kimina_check_response(check_response)
   # parsed_response["sorries"] is available here
   ```

## Potential Enhancements

### Option 1: Add Check to Parser Agents

Modify parser agents to also run a check and get sorries:

```python
# In sketch_parser_agent.py
def _parse_sketch(server_url: str, server_max_retries: int, state: DecomposedFormalTheoremState):
    kimina_client = KiminaClient(api_url=server_url, http_timeout=36000, n_retries=server_max_retries)

    # Get AST
    sketch_with_imports = add_default_imports(str(state["proof_sketch"]))
    ast_code_response = kimina_client.ast_code(sketch_with_imports)
    parsed_ast = parse_kimina_ast_code_response(ast_code_response)
    ast_without_imports = remove_default_imports_from_ast(parsed_ast["ast"])

    # Also get sorries from check
    check_response = kimina_client.check(sketch_with_imports, timeout=36000)
    parsed_check = parse_kimina_check_response(check_response)
    sorries = parsed_check.get("sorries", [])

    # Create AST with sorries
    state["ast"] = AST(ast_without_imports, sorries)

    return {"outputs": [state]}
```

**Pros:**
- Parser immediately has complete type information
- No need to coordinate between agents

**Cons:**
- Runs check twice (once in parser, once in checker)
- Increases latency and server load

### Option 2: Share Sorries Between Agents

Modify checker agents to store sorries in state, then parser uses them:

```python
# In sketch_checker_agent.py
def _check_sketch(...):
    # ... existing check code ...
    parsed_response = parse_kimina_check_response(check_response)

    # Store sorries in state
    state["sorries"] = parsed_response.get("sorries", [])
    state["syntactic"] = parsed_response["pass"]

    return {"outputs": [state]}

# In sketch_parser_agent.py
def _parse_sketch(...):
    # ... existing parse code ...

    # Use sorries if available from previous checker run
    sorries = state.get("sorries", [])
    state["ast"] = AST(ast_without_imports, sorries)

    return {"outputs": [state]}
```

**Pros:**
- No duplicate checks
- Efficient use of server resources
- Agents work together

**Cons:**
- Requires coordination between agents
- Parser depends on checker running first
- Need to handle case where checker hasn't run

### Option 3: Lazy Evaluation

Keep current behavior but document how to add sorries later:

```python
# User code that needs type information
state_with_ast = parse_sketch(...)
state_checked = check_sketch(state_with_ast)

# Recreate AST with sorries when needed
sorries = state_checked.get("sorries", [])
ast_with_types = AST(state_with_ast["ast"].get_ast(), sorries)
subgoal_code = ast_with_types.get_named_subgoal_code("subgoal_name")
```

**Pros:**
- No changes to existing agents
- Maximum flexibility
- Users can choose when to add type info

**Cons:**
- More manual work for users
- Easy to forget

## Recommendation

For immediate use, **Option 3** is already enabled - users can manually add sorries when needed.

For better integration, **Option 2** is recommended:
1. Modify checker agents to store sorries in state
2. Modify parser agents to use sorries from state if available
3. Falls back gracefully if sorries not available

This provides the best balance of efficiency, functionality, and backward compatibility.

## Implementation Example for Option 2

### Step 1: Update sketch_checker_agent.py

```python
def _check_sketch(server_url: str, server_max_retries: int, state: DecomposedFormalTheoremState) -> DecomposedFormalTheoremStates:
    # ... existing code ...
    parsed_response = parse_kimina_check_response(check_response)

    # Store sorries for use by parser (NEW)
    state["sorries"] = parsed_response.get("sorries", [])

    state["syntactic"] = parsed_response["pass"]
    state["errors"] = get_error_str(sketch_with_imports, parsed_response.get("errors", []), False)

    return {"outputs": [state]}
```

### Step 2: Update sketch_parser_agent.py

```python
def _parse_sketch(server_url: str, server_max_retries: int, state: DecomposedFormalTheoremState) -> DecomposedFormalTheoremStates:
    # ... existing code ...
    ast_without_imports = remove_default_imports_from_ast(parsed_response["ast"])

    # Use sorries from checker if available (NEW)
    sorries = state.get("sorries", [])
    state["ast"] = AST(ast_without_imports, sorries)

    return {"outputs": [state]}
```

### Step 3: Update TypedDict definitions

Add `sorries` to state type definitions if needed:

```python
class DecomposedFormalTheoremState(TypedDict, total=False):
    # ... existing fields ...
    sorries: list[dict[str, Any]]  # NEW
```

## Testing Enhanced Integration

After implementing Option 2:

```python
# Test that type information is included
state = {
    "proof_sketch": "theorem test (x : ℕ) : x = x := by\n  have h : x = x := by sorry"
}

# Run checker first (stores sorries in state)
state_checked = _check_sketch(server_url, max_retries, state)

# Run parser (uses sorries from state)
state_parsed = _parse_sketch(server_url, max_retries, state_checked["outputs"][0])

# Extract subgoal with type information
ast = state_parsed["outputs"][0]["ast"]
code = ast.get_named_subgoal_code("h")

# Verify types are included
assert "(x : ℕ)" in code
```
