"""Tests for goedels_poetry.functools module."""

from goedels_poetry.functools import maybe_save


class MockState:
    """Mock state object for testing."""

    def __init__(self) -> None:
        self.save_count = 0

    def save(self) -> None:
        """Mock save method."""
        self.save_count += 1


class ObjectWithMaybeSave:
    """Object with maybe_save decorated methods."""

    def __init__(self, state: MockState):
        self._state = state

    @maybe_save(n=1)
    def method_save_every_call(self, value: int) -> int:
        """Method that saves after every call."""
        return value * 2

    @maybe_save(n=3)
    def method_save_every_third(self, value: int) -> int:
        """Method that saves after every third call."""
        return value + 1

    @maybe_save(n=0)
    def method_no_save(self, value: int) -> int:
        """Method that never saves."""
        return value - 1


def test_maybe_save_every_call() -> None:
    """Test that maybe_save(n=1) saves after every call."""
    state = MockState()
    obj = ObjectWithMaybeSave(state)

    # First call should save
    result = obj.method_save_every_call(5)
    assert result == 10
    assert state.save_count == 1

    # Second call should save
    result = obj.method_save_every_call(3)
    assert result == 6
    assert state.save_count == 2


def test_maybe_save_every_third() -> None:
    """Test that maybe_save(n=3) saves after every third call."""
    state = MockState()
    obj = ObjectWithMaybeSave(state)

    # First call should not save
    obj.method_save_every_third(1)
    assert state.save_count == 0

    # Second call should not save
    obj.method_save_every_third(2)
    assert state.save_count == 0

    # Third call should save
    obj.method_save_every_third(3)
    assert state.save_count == 1

    # Fourth call should not save
    obj.method_save_every_third(4)
    assert state.save_count == 1

    # Fifth call should not save
    obj.method_save_every_third(5)
    assert state.save_count == 1

    # Sixth call should save (second cycle)
    obj.method_save_every_third(6)
    assert state.save_count == 2


def test_maybe_save_disabled() -> None:
    """Test that maybe_save(n=0) never saves."""
    state = MockState()
    obj = ObjectWithMaybeSave(state)

    # Multiple calls should not save
    for i in range(10):
        obj.method_no_save(i)
        assert state.save_count == 0


def test_maybe_save_preserves_return_value() -> None:
    """Test that maybe_save doesn't affect return values."""
    state = MockState()
    obj = ObjectWithMaybeSave(state)

    assert obj.method_save_every_call(7) == 14
    assert obj.method_save_every_third(10) == 11
    assert obj.method_no_save(20) == 19


def test_maybe_save_with_kwargs() -> None:
    """Test that maybe_save works with keyword arguments."""
    state = MockState()

    class ObjWithKwargs:
        def __init__(self, state: MockState):
            self._state = state

        @maybe_save(n=1)
        def method_with_kwargs(self, a: int, b: int = 5) -> int:
            return a + b

    obj = ObjWithKwargs(state)
    assert obj.method_with_kwargs(3) == 8
    assert obj.method_with_kwargs(3, b=7) == 10
    assert state.save_count == 2


def test_maybe_save_independent_counters() -> None:
    """Test that different decorated methods have independent counters."""
    # Note: Counters are stored on the function object itself, so they're shared
    # across all instances. We need to use fresh objects to test independence.

    # Test method_save_every_call (n=1) on its own object
    state1 = MockState()
    obj1 = ObjectWithMaybeSave(state1)
    obj1.method_save_every_call(5)
    assert state1.save_count == 1

    # Test method_save_every_third (n=3) on its own object
    state2 = MockState()
    obj2 = ObjectWithMaybeSave(state2)
    obj2.method_save_every_third(1)
    obj2.method_save_every_third(2)
    obj2.method_save_every_third(3)
    assert state2.save_count == 1  # Saves on third call
