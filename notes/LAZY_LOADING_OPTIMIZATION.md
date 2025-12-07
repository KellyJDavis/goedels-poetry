# Lazy Loading Optimization

## Overview

Implemented lazy loading for `FORMALIZER_AGENT_LLM` and `SEMANTICS_AGENT_LLM` to significantly speed up startup when processing formal theorems.

## Problem

Previously, both the formalizer and semantics LLMs were eagerly loaded at module import time in `goedels_poetry/config/llm.py`. This meant:

1. **Every** invocation of the CLI would initialize these large Ollama models (~32B and ~30B parameters)
2. **Formal theorem processing** would wait unnecessarily since these models are only used for informal theorems
3. Startup time included ~21 seconds of unnecessary model initialization

## Solution

### Changes Made

#### 1. `goedels_poetry/config/llm.py`
- Created lazy-loading functions: `get_formalizer_agent_llm()` and `get_semantics_agent_llm()`
- These functions only create LLMs on first access (models must be pre-downloaded)
- Results are cached in module-level variables for reuse
- Removed eager initialization of these LLMs at import time

#### 2. `goedels_poetry/framework.py`
- Updated `GoedelsPoetryConfig` to accept optional LLM parameters (default `None`)
- Added `@property` decorators for `formalizer_agent_llm` and `semantics_agent_llm`
- Properties call the lazy-loading functions on first access
- Prover and decomposer LLMs remain eagerly loaded (needed for all workflows)

#### 3. `goedels_poetry/state.py`
- Added comment clarifying why these LLMs are not imported
- No functional changes needed

## Results

### Performance Improvement

**Formal Theorem Processing:**
- **Before:** ~21 seconds to load formalizer + semantics LLMs on every invocation
- **After:** 0 seconds (LLMs not loaded at all)
- **Speedup:** Immediate startup (~21 second improvement)

**Informal Theorem Processing:**
- **Before:** ~21 seconds to load LLMs at import time
- **After:** ~21 seconds to load LLMs on first access (when needed)
- **Speedup:** No change (models still needed, just loaded later)

### Testing

Verified that:
1. ✅ Formal theorem workflow does NOT load formalizer/semantics LLMs
2. ✅ Informal theorem workflow DOES load them (lazily, when accessed)
3. ✅ All 99 existing tests pass without modification
4. ✅ LLMs are cached after first load for efficiency

## Usage Flow

### Formal Theorems
```python
# User runs: goedels-poetry --formal-theorem "theorem foo : 1 + 1 = 2 := by sorry"

# 1. Import llm.py (fast - no formalizer/semantics loading)
# 2. Create GoedelsPoetryConfig() (fast - uses defaults)
# 3. Process formal theorem (uses prover/decomposer only)
# Result: Start proving immediately!
```

### Informal Theorems
```python
# User runs: goedels-poetry --informal-theorem "Prove 1 + 1 = 2"

# 1. Import llm.py (fast - no formalizer/semantics loading yet)
# 2. Create GoedelsPoetryConfig() (fast)
# 3. Call framework.formalize_informal_theorem()
#    -> Accesses config.formalizer_agent_llm
#    -> Triggers lazy load (~13s to init)
# 4. Call framework.check_informal_theorem_semantics()
#    -> Accesses config.semantics_agent_llm
#    -> Triggers lazy load (~8s to init)
# 5. Continue with proof...
# Result: Models loaded only when actually needed!
```

## Backward Compatibility

The changes are fully backward compatible:
- Existing code that accesses `config.formalizer_agent_llm` will work identically
- The properties ensure transparent lazy loading
- Tests require no modifications
- CLI usage remains unchanged

## Future Considerations

Could extend this pattern to:
- Make `DECOMPOSER_AGENT_LLM` lazy-loadable (only needed for complex proofs requiring decomposition)
- Add environment variable to force eager loading if desired
- Add metrics to track actual model usage patterns
