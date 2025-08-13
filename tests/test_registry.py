"""Tests for check registry system."""

import pytest
import asyncio
from typing import Any
from flex_evals.registry import (
    CheckRegistry, register, get_check_class, get_check_info,
    is_async_check, list_registered_checks, clear_registry,
    get_registry_state, restore_registry_state, get_latest_version, list_versions,
    get_version_for_class, get_check_type_for_class, _global_registry,
)
from flex_evals.checks.base import BaseCheck, BaseAsyncCheck, EvaluationContext
from flex_evals import CheckType, Output, TestCase
from tests.conftest import restore_standard_checks


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

    def test_register_multiple_versions(self):
        """Test multiple version registration behavior."""
        # First registration
        self.registry.register("test_check", SampleSyncCheck, "1.0.0")

        # Same version - should allow re-registration
        self.registry.register("test_check", SampleSyncCheck, "1.0.0")

        # Different version - should now be allowed (new behavior)
        self.registry.register("test_check", SampleSyncCheck, "2.0.0")

        # Verify both versions exist
        v1_class = self.registry.get_check_class("test_check", "1.0.0")
        v2_class = self.registry.get_check_class("test_check", "2.0.0")
        assert v1_class == SampleSyncCheck
        assert v2_class == SampleSyncCheck

        # Latest should return v2.0.0
        latest_class = self.registry.get_check_class("test_check")
        assert latest_class == SampleSyncCheck

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

        # Verify nested structure: {check_type: {version: info}}
        assert "1.0.0" in checks["check1"]
        assert checks["check1"]["1.0.0"]["class"] == SampleSyncCheck
        assert checks["check1"]["1.0.0"]["version"] == "1.0.0"
        assert checks["check1"]["1.0.0"]["is_async"] is False

        assert "2.0.0" in checks["check2"]
        assert checks["check2"]["2.0.0"]["class"] == SampleAsyncCheck
        assert checks["check2"]["2.0.0"]["version"] == "2.0.0"
        assert checks["check2"]["2.0.0"]["is_async"] is True

    def test_registry_clear(self):
        """Test clearing registry."""
        self.registry.register("test_check", SampleSyncCheck)
        assert len(self.registry.list_registered_checks()) == 1

        self.registry.clear()
        assert len(self.registry.list_registered_checks()) == 0

        with pytest.raises(ValueError, match="is not registered"):
            self.registry.get_check_class("test_check")

    def test_get_latest_version(self):
        """Test getting latest version using semantic versioning."""
        # Register versions in non-sorted order
        self.registry.register("version_test", SampleSyncCheck, "1.0.0")
        self.registry.register("version_test", SampleSyncCheck, "2.1.0")
        self.registry.register("version_test", SampleSyncCheck, "1.10.0")
        self.registry.register("version_test", SampleSyncCheck, "2.0.0")

        latest = self.registry.get_latest_version("version_test")
        assert latest == "2.1.0"

        # Latest should be returned by default
        latest_class = self.registry.get_check_class("version_test")
        explicit_latest = self.registry.get_check_class("version_test", "2.1.0")
        assert latest_class == explicit_latest

    def test_list_versions_sorted(self):
        """Test listing versions in sorted order."""
        # Register versions in non-sorted order
        versions = ["1.0.0", "2.1.0", "1.10.0", "2.0.0"]
        for version in versions:
            self.registry.register("sort_test", SampleSyncCheck, version)

        sorted_versions = self.registry.list_versions("sort_test")
        expected = ["1.0.0", "1.10.0", "2.0.0", "2.1.0"]
        assert sorted_versions == expected

    def test_version_specific_retrieval(self):
        """Test retrieving specific versions."""
        class CheckV1(BaseCheck):
            def __call__(self, arguments: dict[str, Any], context: EvaluationContext) -> dict[str, Any]:  # noqa: ARG002, E501
                return {"passed": True, "version": "1.0.0"}

        class CheckV2(BaseCheck):
            def __call__(self, arguments: dict[str, Any], context: EvaluationContext) -> dict[str, Any]:  # noqa: ARG002, E501
                return {"passed": True, "version": "2.0.0"}

        self.registry.register("multi_version", CheckV1, "1.0.0")
        self.registry.register("multi_version", CheckV2, "2.0.0")

        # Test specific version retrieval
        v1_class = self.registry.get_check_class("multi_version", "1.0.0")
        v2_class = self.registry.get_check_class("multi_version", "2.0.0")
        latest_class = self.registry.get_check_class("multi_version")

        assert v1_class == CheckV1
        assert v2_class == CheckV2
        assert latest_class == CheckV2  # Should be latest (2.0.0)

        # Test info retrieval
        v1_info = self.registry.get_check_info("multi_version", "1.0.0")
        v2_info = self.registry.get_check_info("multi_version", "2.0.0")

        assert v1_info["version"] == "1.0.0"
        assert v2_info["version"] == "2.0.0"

    def test_version_error_cases(self):
        """Test error cases for versioned registry."""
        self.registry.register("error_test", SampleSyncCheck, "1.0.0")

        # Test non-existent version
        with pytest.raises(ValueError, match="Version '2.0.0' not found for check type 'error_test'"):  # noqa: E501
            self.registry.get_check_class("error_test", "2.0.0")

        # Test non-existent check type for version functions
        with pytest.raises(ValueError, match="Check type 'nonexistent' is not registered"):
            self.registry.get_latest_version("nonexistent")

        with pytest.raises(ValueError, match="Check type 'nonexistent' is not registered"):
            self.registry.list_versions("nonexistent")


class TestRegistryDecorator:
    """Test registry decorator functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear global registry for clean tests
        clear_registry()

    def teardown_method(self):
        """Clean up after tests."""
        clear_registry()
        # Restore standard checks for other tests
        restore_standard_checks()

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

        # Registering different version should now be allowed
        @register("conflict_test", version="2.0.0")
        class SecondCheck(BaseCheck):
            def __call__(self, arguments, context):  # noqa: ANN001, ARG002
                return {"passed": False}

        # Both versions should be available
        v1_class = get_check_class("conflict_test", "1.0.0")
        v2_class = get_check_class("conflict_test", "2.0.0")
        latest_class = get_check_class("conflict_test")

        assert v1_class == FirstCheck
        assert v2_class == SecondCheck
        assert latest_class == SecondCheck  # Latest should be 2.0.0

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

        # Different version should now be allowed
        @register(CheckType.EXACT_MATCH, version="2.0.0")
        class TestCheck3(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True}

        # Verify multiple versions exist
        get_check_class('exact_match', '1.0.0')
        v2_class = get_check_class('exact_match', '2.0.0')
        latest_class = get_check_class('exact_match')

        assert v2_class == TestCheck3
        assert latest_class == TestCheck3  # Latest should be 2.0.0

    def test_register_async_check_with_enum(self):
        """Test registering async check with enum."""
        @register(CheckType.SEMANTIC_SIMILARITY, version="1.0.0")
        class TestAsyncCheck(BaseAsyncCheck):
            async def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"score": 0.9}

        check_class = get_check_class('semantic_similarity')
        assert check_class == TestAsyncCheck
        assert is_async_check('semantic_similarity')

    def test_module_level_versioning_functions(self):
        """Test module-level version management functions."""
        @register("version_func_test", version="1.0.0")
        class TestCheck_v1(BaseCheck):  # noqa: N801
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True, "version": "1.0.0"}

        @register("version_func_test", version="1.5.0")
        class TestCheck_v1_5(BaseCheck):  # noqa: N801
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True, "version": "1.5.0"}

        @register("version_func_test", version="2.0.0")
        class TestCheck_v2(BaseCheck):  # noqa: N801
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True, "version": "2.0.0"}

        # Test get_latest_version
        latest = get_latest_version("version_func_test")
        assert latest == "2.0.0"

        # Test list_versions (should be sorted)
        versions = list_versions("version_func_test")
        assert versions == ["1.0.0", "1.5.0", "2.0.0"]

        # Test version-aware retrieval
        v1_class = get_check_class("version_func_test", "1.0.0")
        v2_class = get_check_class("version_func_test", "2.0.0")
        latest_class = get_check_class("version_func_test")

        assert v1_class == TestCheck_v1
        assert v2_class == TestCheck_v2
        assert latest_class == TestCheck_v2

        # Test version-aware info
        v1_info = get_check_info("version_func_test", "1.0.0")
        v2_info = get_check_info("version_func_test", "2.0.0")
        latest_info = get_check_info("version_func_test")

        assert v1_info["version"] == "1.0.0"
        assert v2_info["version"] == "2.0.0"
        assert latest_info["version"] == "2.0.0"

        # Test version-aware async check
        assert not is_async_check("version_func_test", "1.0.0")
        assert not is_async_check("version_func_test", "2.0.0")
        assert not is_async_check("version_func_test")  # Latest


class TestRegistryStateSerialization:
    """Test registry state serialization/deserialization functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Clear global registry for clean tests
        clear_registry()

    def teardown_method(self):
        """Clean up after tests."""
        clear_registry()
        # Restore standard checks for other tests
        restore_standard_checks()

    def test_get_registry_state_empty(self):
        """Test getting state from empty registry."""
        state = get_registry_state()
        assert isinstance(state, dict)
        assert len(state) == 0

    def test_get_registry_state_with_checks(self):
        """Test getting state with registered checks."""
        @register("test_sync", version="1.0.0")
        class TestSyncCheck(BaseCheck):
            def __call__(self, **kwargs) -> dict[str, Any]:  # noqa
                return {"passed": True}

        @register("test_async", version="2.0.0")
        class TestAsyncCheck(BaseAsyncCheck):
            async def __call__(self, **kwargs) -> dict[str, Any]:  # noqa
                return {"passed": True}

        state = get_registry_state()

        # Verify structure
        assert isinstance(state, dict)
        assert len(state) == 2
        assert "test_sync" in state
        assert "test_async" in state

        # Verify sync check state (nested structure)
        sync_versions = state["test_sync"]
        assert "1.0.0" in sync_versions
        sync_state = sync_versions["1.0.0"]
        assert sync_state["class"] == TestSyncCheck
        assert sync_state["version"] == "1.0.0"
        assert sync_state["is_async"] is False

        # Verify async check state (nested structure)
        async_versions = state["test_async"]
        assert "2.0.0" in async_versions
        async_state = async_versions["2.0.0"]
        assert async_state["class"] == TestAsyncCheck
        assert async_state["version"] == "2.0.0"
        assert async_state["is_async"] is True

    def test_restore_registry_state_empty(self):
        """Test restoring empty registry state."""
        # Register some checks first
        @register("temp_check")
        class TempCheck(BaseCheck):
            def __call__(self, **kwargs) -> dict[str, Any]:  # noqa
                return {"passed": True}

        # Verify check is registered
        assert len(list_registered_checks()) == 1

        # Restore empty state
        restore_registry_state({})

        # Registry should be empty
        assert len(list_registered_checks()) == 0
        with pytest.raises(ValueError, match="is not registered"):
            get_check_class("temp_check")

    def test_restore_registry_state_with_checks(self):
        """Test restoring registry state with checks."""
        # Create test check classes
        class RestoredSyncCheck(BaseCheck):
            def __call__(self, **kwargs) -> dict[str, Any]:  # noqa
                return {"passed": True, "type": "sync"}

        class RestoredAsyncCheck(BaseAsyncCheck):
            async def __call__(self, **kwargs) -> dict[str, Any]:  # noqa
                return {"passed": True, "type": "async"}

        # Create state to restore (nested structure)
        state_to_restore = {
            "restored_sync": {
                "1.5.0": {
                    "class": RestoredSyncCheck,
                    "version": "1.5.0",
                    "is_async": False,
                },
            },
            "restored_async": {
                "2.5.0": {
                    "class": RestoredAsyncCheck,
                    "version": "2.5.0",
                    "is_async": True,
                },
            },
        }

        # Restore the state
        restore_registry_state(state_to_restore)

        # Verify checks are registered correctly
        assert len(list_registered_checks()) == 2

        # Test sync check
        sync_class = get_check_class("restored_sync")
        assert sync_class == RestoredSyncCheck
        assert is_async_check("restored_sync") is False

        sync_info = get_check_info("restored_sync")
        assert sync_info["version"] == "1.5.0"
        assert sync_info["is_async"] is False

        # Test async check
        async_class = get_check_class("restored_async")
        assert async_class == RestoredAsyncCheck
        assert is_async_check("restored_async") is True

        async_info = get_check_info("restored_async")
        assert async_info["version"] == "2.5.0"
        assert async_info["is_async"] is True

    def test_get_and_restore_registry_state_roundtrip(self):
        """Test complete roundtrip: get state, clear, restore state."""
        # Register original checks
        @register("original_sync", version="1.0.0")
        class OriginalSyncCheck(BaseCheck):
            def __call__(self, test_value: str = "default", **kwargs) -> dict[str, Any]:  # noqa
                return {"passed": True, "test_value": test_value}

        @register("original_async", version="2.0.0")
        class OriginalAsyncCheck(BaseAsyncCheck):
            async def __call__(self, delay: float = 0.01, **kwargs) -> dict[str, Any]:  # noqa
                await asyncio.sleep(delay)
                return {"passed": True, "delay": delay}

        # Capture original state
        original_state = get_registry_state()
        assert len(original_state) == 2

        # Clear registry
        clear_registry()
        assert len(list_registered_checks()) == 0

        # Restore original state
        restore_registry_state(original_state)

        # Verify everything is restored correctly
        assert len(list_registered_checks()) == 2

        # Test sync check functionality
        sync_class = get_check_class("original_sync")
        assert sync_class == OriginalSyncCheck

        sync_instance = sync_class()
        sync_result = sync_instance(test_value="custom_value")
        assert sync_result["passed"] is True
        assert sync_result["test_value"] == "custom_value"

        # Test async check functionality
        async_class = get_check_class("original_async")
        assert async_class == OriginalAsyncCheck

        # Verify metadata is preserved
        sync_info = get_check_info("original_sync")
        assert sync_info["version"] == "1.0.0"
        assert sync_info["is_async"] is False

        async_info = get_check_info("original_async")
        assert async_info["version"] == "2.0.0"
        assert async_info["is_async"] is True

    def test_restore_registry_state_clears_existing(self):
        """Test that restore_registry_state clears existing registrations."""
        # Register initial checks
        @register("initial_check")
        class InitialCheck(BaseCheck):
            def __call__(self, **kwargs) -> dict[str, Any]:  # noqa
                return {"passed": True}

        assert len(list_registered_checks()) == 1
        assert "initial_check" in list_registered_checks()

        # Create new state with different checks (nested structure)
        class NewCheck(BaseCheck):
            def __call__(self, **kwargs) -> dict[str, Any]:  # noqa
                return {"passed": False}

        new_state = {
            "new_check": {
                "3.0.0": {
                    "class": NewCheck,
                    "version": "3.0.0",
                    "is_async": False,
                },
            },
        }

        # Restore new state
        restore_registry_state(new_state)

        # Old check should be gone, new check should be present
        assert len(list_registered_checks()) == 1
        assert "new_check" in list_registered_checks()
        assert "initial_check" not in list_registered_checks()

        with pytest.raises(ValueError, match="is not registered"):
            get_check_class("initial_check")

        # New check should work
        new_class = get_check_class("new_check")
        assert new_class == NewCheck

    def test_registry_state_serialization_types(self):
        """Test that registry state contains proper types for serialization."""
        @register("type_test", version="1.2.3")
        class TypeTestCheck(BaseCheck):
            def __call__(self, **kwargs) -> dict[str, Any]:  # noqa
                return {"passed": True}

        state = get_registry_state()

        # Verify all values are serializable types (nested structure)
        assert isinstance(state, dict)
        assert isinstance(state["type_test"], dict)
        assert "1.2.3" in state["type_test"]
        version_state = state["type_test"]["1.2.3"]
        assert isinstance(version_state["version"], str)
        assert isinstance(version_state["is_async"], bool)

        # Class should be the actual class object (this is what gets serialized)
        assert version_state["class"] == TypeTestCheck

    def test_restore_registry_state_with_enum_types(self):
        """Test restoring state with CheckType enum compatibility."""
        class EnumTestCheck(BaseCheck):
            def __call__(self, **kwargs) -> dict[str, Any]:  # noqa
                return {"passed": True}

        # Create state using string key (how it's stored internally) with nested structure
        state_with_enum = {
            "exact_match": {  # CheckType.EXACT_MATCH string value
                "1.0.0": {
                    "class": EnumTestCheck,
                    "version": "1.0.0",
                    "is_async": False,
                },
            },
        }

        restore_registry_state(state_with_enum)

        # Should be retrievable by string
        check_class = get_check_class("exact_match")
        assert check_class == EnumTestCheck


class TestRegistryReverseMapping:
    """Test registry reverse mapping functionality (class -> version/type)."""

    def setup_method(self):
        """Set up test fixtures."""
        clear_registry()

    def teardown_method(self):
        """Clean up after tests."""
        clear_registry()
        restore_standard_checks()

    def test_get_version_for_class_basic(self):
        """Test getting version for registered class."""
        @register("version_lookup_test", version="2.1.0")
        class VersionLookupTest(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True}

        # Should be able to get version from class
        version = get_version_for_class(VersionLookupTest)
        assert version == "2.1.0"

        # Should also work with registry method
        version_registry = _global_registry.get_version_for_class(VersionLookupTest)
        assert version_registry == "2.1.0"

    def test_get_check_type_for_class_basic(self):
        """Test getting check type for registered class."""
        @register("type_lookup_test", version="1.0.0")
        class TypeLookupTest(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True}

        # Should be able to get check type from class
        check_type = get_check_type_for_class(TypeLookupTest)
        assert check_type == "type_lookup_test"

        # Should also work with registry method
        check_type_registry = _global_registry.get_check_type_for_class(TypeLookupTest)
        assert check_type_registry == "type_lookup_test"

    def test_reverse_mapping_with_multiple_versions(self):
        """Test reverse mapping works correctly with multiple versions of same type."""
        class CheckV1(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"version": "1.0.0"}

        class CheckV2(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"version": "2.0.0"}

        register("multi_version_test", version="1.0.0")(CheckV1)
        register("multi_version_test", version="2.0.0")(CheckV2)

        # Each class should map to its specific version
        assert get_version_for_class(CheckV1) == "1.0.0"
        assert get_version_for_class(CheckV2) == "2.0.0"

        # Both should map to same check type
        assert get_check_type_for_class(CheckV1) == "multi_version_test"
        assert get_check_type_for_class(CheckV2) == "multi_version_test"

    def test_reverse_mapping_with_async_checks(self):
        """Test reverse mapping works with async checks."""
        @register("async_reverse_test", version="1.5.0")
        class AsyncReverseTest(BaseAsyncCheck):
            async def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True}

        assert get_version_for_class(AsyncReverseTest) == "1.5.0"
        assert get_check_type_for_class(AsyncReverseTest) == "async_reverse_test"

    def test_reverse_mapping_with_enum_types(self):
        """Test reverse mapping works with CheckType enums."""
        @register(CheckType.EXACT_MATCH, version="3.0.0")
        class EnumReverseTest(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True}

        # Should work with enum registration
        assert get_version_for_class(EnumReverseTest) == "3.0.0"
        assert get_check_type_for_class(EnumReverseTest) == "exact_match"  # String form

    def test_reverse_mapping_error_cases(self):
        """Test error cases for reverse mapping."""
        class UnregisteredCheck(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True}

        # Should raise ValueError for unregistered class
        with pytest.raises(ValueError, match="Class .* is not registered"):
            get_version_for_class(UnregisteredCheck)

        with pytest.raises(ValueError, match="Class .* is not registered"):
            get_check_type_for_class(UnregisteredCheck)

        # Should also raise for registry methods
        with pytest.raises(ValueError, match="Class .* is not registered"):
            _global_registry.get_version_for_class(UnregisteredCheck)

        with pytest.raises(ValueError, match="Class .* is not registered"):
            _global_registry.get_check_type_for_class(UnregisteredCheck)

    def test_reverse_mapping_with_re_registration(self):
        """Test reverse mapping updates correctly with re-registration."""
        class ReRegisteredCheck(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True}

        # Register first time
        register("re_register_test", version="1.0.0")(ReRegisteredCheck)
        assert get_version_for_class(ReRegisteredCheck) == "1.0.0"

        # Re-register same version (should work)
        register("re_register_test", version="1.0.0")(ReRegisteredCheck)
        assert get_version_for_class(ReRegisteredCheck) == "1.0.0"

        # Register with different version (should update)
        register("re_register_test", version="2.0.0")(ReRegisteredCheck)
        assert get_version_for_class(ReRegisteredCheck) == "2.0.0"
        assert get_check_type_for_class(ReRegisteredCheck) == "re_register_test"

    def test_reverse_mapping_cleared_with_registry(self):
        """Test reverse mapping is cleared when registry is cleared."""
        @register("clear_test", version="1.0.0")
        class ClearTest(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True}

        # Should work before clear
        assert get_version_for_class(ClearTest) == "1.0.0"

        # Clear registry
        clear_registry()

        # Should no longer work after clear
        with pytest.raises(ValueError, match="Class .* is not registered"):
            get_version_for_class(ClearTest)

    def test_reverse_mapping_consistency(self):
        """Test that forward and reverse mappings are consistent."""
        @register("consistency_test", version="1.2.3")
        class ConsistencyTest(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True}

        # Forward mapping
        forward_class = get_check_class("consistency_test", "1.2.3")
        forward_info = get_check_info("consistency_test", "1.2.3")

        # Reverse mapping
        reverse_version = get_version_for_class(ConsistencyTest)
        reverse_type = get_check_type_for_class(ConsistencyTest)

        # Should be consistent
        assert forward_class == ConsistencyTest
        assert forward_info["version"] == reverse_version == "1.2.3"
        assert reverse_type == "consistency_test"

    def test_reverse_mapping_with_different_check_types(self):
        """Test reverse mapping works correctly with different check types."""
        @register("type_a", version="1.0.0")
        class CheckTypeA(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"type": "A"}

        @register("type_b", version="1.0.0")
        class CheckTypeB(BaseAsyncCheck):
            async def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"type": "B"}

        # Each should map to its own type and version
        assert get_version_for_class(CheckTypeA) == "1.0.0"
        assert get_check_type_for_class(CheckTypeA) == "type_a"

        assert get_version_for_class(CheckTypeB) == "1.0.0"
        assert get_check_type_for_class(CheckTypeB) == "type_b"

    def test_reverse_mapping_integration_with_existing_registry(self):
        """Test reverse mapping integrates properly with all existing registry features."""
        @register("integration_test", version="1.0.0")
        class IntegrationTestV1(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"version": "1.0.0"}

        @register("integration_test", version="2.0.0")
        class IntegrationTestV2(BaseAsyncCheck):
            async def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"version": "2.0.0"}

        # Test all registry functions still work
        assert len(list_registered_checks()) == 1
        assert "integration_test" in list_registered_checks()
        assert get_latest_version("integration_test") == "2.0.0"
        assert list_versions("integration_test") == ["1.0.0", "2.0.0"]
        assert is_async_check("integration_test", "1.0.0") is False
        assert is_async_check("integration_test", "2.0.0") is True

        # Test reverse mapping works
        assert get_version_for_class(IntegrationTestV1) == "1.0.0"
        assert get_check_type_for_class(IntegrationTestV1) == "integration_test"
        assert get_version_for_class(IntegrationTestV2) == "2.0.0"
        assert get_check_type_for_class(IntegrationTestV2) == "integration_test"

        # Test state serialization/restoration preserves reverse mapping
        state = get_registry_state()
        clear_registry()

        # Should fail after clear
        with pytest.raises(ValueError, match="Class .* is not registered"):
            get_version_for_class(IntegrationTestV1)

        # Restore and test again
        restore_registry_state(state)
        assert get_version_for_class(IntegrationTestV1) == "1.0.0"
        assert get_version_for_class(IntegrationTestV2) == "2.0.0"
