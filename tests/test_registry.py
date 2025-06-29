"""Tests for check registry system."""

import pytest
from typing import Any

from flex_evals.registry import (
    CheckRegistry, register, get_check_class, get_check_info,
    is_async_check, list_registered_checks, clear_registry,
)
from flex_evals.checks.base import BaseCheck, BaseAsyncCheck, EvaluationContext
from flex_evals.constants import CheckType
from flex_evals.schemas.output import Output
from flex_evals.schemas.test_case import TestCase


class SampleSyncCheck(BaseCheck):
    """Sample sync check for testing."""

    def __call__(self, arguments: dict[str, Any], context: EvaluationContext) -> dict[str, Any]:  # noqa: ARG002
        return {"passed": True}


class SampleAsyncCheck(BaseAsyncCheck):
    """Sample async check for testing."""

    async def __call__(self, arguments: dict[str, Any], context: EvaluationContext) -> dict[str, Any]:  # noqa: ARG002, E501
        return {"passed": True}


class TestCheckRegistry:
    """Test CheckRegistry functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.registry = CheckRegistry()

    def test_register_basic(self):
        """Test basic check registration."""
        self.registry.register("test_check", SampleSyncCheck, "1.0.0")

        # Test check is registered
        check_class = self.registry.get_check_class("test_check")
        assert check_class == SampleSyncCheck

        # Test check info
        info = self.registry.get_check_info("test_check")
        assert info["class"] == SampleSyncCheck
        assert info["version"] == "1.0.0"
        assert info["is_async"] is False

    def test_register_with_version(self):
        """Test version specification in registration."""
        self.registry.register("test_check", SampleSyncCheck, "2.1.0")

        info = self.registry.get_check_info("test_check")
        assert info["version"] == "2.1.0"

    def test_register_conflict_handling(self):
        """Test duplicate registration behavior."""
        # First registration
        self.registry.register("test_check", SampleSyncCheck, "1.0.0")

        # Same version - should allow re-registration
        self.registry.register("test_check", SampleSyncCheck, "1.0.0")

        # Different version - should raise error
        with pytest.raises(ValueError, match="already registered with version 1.0.0"):
            self.registry.register("test_check", SampleSyncCheck, "2.0.0")

    def test_get_check_class_success(self):
        """Test successful check class retrieval."""
        self.registry.register("sample_sync", SampleSyncCheck)
        self.registry.register("sample_async", SampleAsyncCheck)

        sync_class = self.registry.get_check_class("sample_sync")
        async_class = self.registry.get_check_class("sample_async")

        assert sync_class == SampleSyncCheck
        assert async_class == SampleAsyncCheck

    def test_get_check_class_not_found(self):
        """Test error for unregistered check type."""
        with pytest.raises(ValueError, match="Check type 'nonexistent' is not registered"):
            self.registry.get_check_class("nonexistent")

    def test_is_async_check_detection(self):
        """Test async check detection."""
        self.registry.register("sync_check", SampleSyncCheck)
        self.registry.register("async_check", SampleAsyncCheck)

        assert self.registry.is_async_check("sync_check") is False
        assert self.registry.is_async_check("async_check") is True

    def test_list_registered_checks(self):
        """Test listing all registered checks."""
        self.registry.register("check1", SampleSyncCheck, "1.0.0")
        self.registry.register("check2", SampleAsyncCheck, "2.0.0")

        checks = self.registry.list_registered_checks()

        assert len(checks) == 2
        assert "check1" in checks
        assert "check2" in checks

        assert checks["check1"]["class"] == SampleSyncCheck
        assert checks["check1"]["version"] == "1.0.0"
        assert checks["check1"]["is_async"] is False

        assert checks["check2"]["class"] == SampleAsyncCheck
        assert checks["check2"]["version"] == "2.0.0"
        assert checks["check2"]["is_async"] is True

    def test_registry_clear(self):
        """Test clearing registry."""
        self.registry.register("test_check", SampleSyncCheck)
        assert len(self.registry.list_registered_checks()) == 1

        self.registry.clear()
        assert len(self.registry.list_registered_checks()) == 0

        with pytest.raises(ValueError, match="is not registered"):
            self.registry.get_check_class("test_check")


class TestRegistryDecorator:
    """Test registry decorator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear global registry for clean tests
        clear_registry()

    def teardown_method(self):
        """Clean up after tests."""
        clear_registry()

    def test_register_decorator_basic(self):
        """Test basic check registration with decorator."""
        @register("decorated_check", version="1.0.0")
        class DecoratedCheck(BaseCheck):
            def __call__(self, arguments, context):  # noqa: ANN001, ARG002
                return {"passed": True}

        # Test check is registered in global registry
        check_class = get_check_class("decorated_check")
        assert check_class == DecoratedCheck

        info = get_check_info("decorated_check")
        assert info["version"] == "1.0.0"
        assert info["is_async"] is False

    def test_register_async_check(self):
        """Test registering async check with decorator."""
        @register("decorated_async", version="2.0.0")
        class DecoratedAsyncCheck(BaseAsyncCheck):
            async def __call__(self, arguments, context):  # noqa: ANN001, ARG002
                return {"passed": True}

        assert is_async_check("decorated_async") is True

        info = get_check_info("decorated_async")
        assert info["version"] == "2.0.0"
        assert info["is_async"] is True

    def test_register_custom_check(self):
        """Test registering user-defined checks."""
        @register("custom_business_logic", version="1.2.3")
        class CustomBusinessCheck(BaseCheck):
            def __call__(self, arguments, context):  # noqa: ANN001, ARG002
                # Custom business logic
                business_value = arguments.get("business_value", 0)
                threshold = arguments.get('threshold', 100)

                return {
                    "passed": business_value > threshold,
                    "business_value": business_value,
                    'threshold': threshold,
                }

        check_class = get_check_class("custom_business_logic")
        assert check_class == CustomBusinessCheck

        # Test the check works
        check_instance = check_class()
        test_case = TestCase(id="test", input="test")
        output = Output(value="test")
        context = EvaluationContext(test_case, output)

        # Test arguments are properly passed
        result = check_instance({"business_value": 150, 'threshold': 100}, context)
        assert result["passed"] is True
        assert result["business_value"] == 150
        assert result['threshold'] == 100

    def test_global_registry_functions(self):
        """Test global registry access functions."""
        @register("func_test_check")
        class FuncTestCheck(BaseCheck):
            def __call__(self, arguments, context):  # noqa: ANN001, ARG002
                return {"passed": True}

        # Test global functions work
        assert get_check_class("func_test_check") == FuncTestCheck
        assert is_async_check("func_test_check") is False

        checks = list_registered_checks()
        assert "func_test_check" in checks

        # Test clear function
        clear_registry()
        with pytest.raises(ValueError):  # noqa: PT011
            get_check_class("func_test_check")

    def test_version_conflict_in_decorator(self):
        """Test version conflict handling with decorator."""
        @register("conflict_test", version="1.0.0")
        class FirstCheck(BaseCheck):
            def __call__(self, arguments, context):  # noqa: ANN001, ARG002
                return {"passed": True}

        # Attempt to register same type with different version should fail
        with pytest.raises(ValueError, match="already registered with version 1.0.0"):
            @register("conflict_test", version="2.0.0")
            class SecondCheck(BaseCheck):
                def __call__(self, arguments, context):  # noqa: ANN001, ARG002
                    return {"passed": False}

    def test_decorator_preserves_class(self):
        """Test decorator returns the original class unchanged."""
        @register("preserve_test")
        class PreserveTest(BaseCheck):
            def __call__(self, arguments, context):  # noqa: ANN001, ARG002
                return {"test": "value"}

            def custom_method(self) -> str:
                return "custom"

        # Class should be unchanged
        assert hasattr(PreserveTest, "custom_method")
        assert PreserveTest().custom_method() == "custom"

        # But still registered
        assert get_check_class("preserve_test") == PreserveTest

    def test_register_with_check_type_enum(self):
        """Test registering check with CheckType enum."""
        @register(CheckType.EXACT_MATCH, version="1.0.0")
        class TestCheck(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True}

        # Should be retrievable by string
        check_class = get_check_class('exact_match')
        assert check_class == TestCheck

    def test_register_with_string_and_enum_compatibility(self):
        """Test that enum and string registrations are compatible."""
        @register(CheckType.EXACT_MATCH, version="1.0.0")
        class TestCheck1(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True}

        # This should work (same version, equivalent types)
        @register('exact_match', version="1.0.0")
        class TestCheck2(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": False}

        # Different version should raise error
        with pytest.raises(ValueError, match="already registered"):
            @register(CheckType.EXACT_MATCH, version="2.0.0")
            class TestCheck3(BaseCheck):
                def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                    return {"passed": True}

    def test_register_async_check_with_enum(self):
        """Test registering async check with enum."""
        @register(CheckType.SEMANTIC_SIMILARITY, version="1.0.0")
        class TestAsyncCheck(BaseAsyncCheck):
            async def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"score": 0.9}

        check_class = get_check_class('semantic_similarity')
        assert check_class == TestAsyncCheck
        assert is_async_check('semantic_similarity')
