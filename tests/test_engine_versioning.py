"""Tests for engine version-aware check execution."""


from flex_evals.engine import evaluate
from flex_evals.schemas import TestCase, Output, Check
from flex_evals.registry import register, clear_registry
from flex_evals.checks.base import BaseCheck, BaseAsyncCheck
from tests.conftest import restore_standard_checks
from typing import Any


class TestEngineVersioning:
    """Test engine handling of versioned checks."""

    def setup_method(self):
        """Clear registry before each test."""
        clear_registry()

    def teardown_method(self):
        """Restore standard checks after each test."""
        clear_registry()
        restore_standard_checks()

    def test_engine_uses_specific_check_version(self):
        """Test that engine uses the specific version requested in Check object."""

        @register("version_test", version="1.0.0")
        class VersionTest_v1(BaseCheck):  # noqa: N801
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True, "version": "1.0.0"}

        @register("version_test", version="2.0.0")
        class VersionTest_v2(BaseCheck):  # noqa: N801
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True, "version": "2.0.0"}

        # Create test case and output
        test_case = TestCase(id="test1", input="test input")
        output = Output(value="test output")

        # Create checks with specific versions
        check_v1 = Check(type="version_test", arguments={}, version="1.0.0")
        check_v2 = Check(type="version_test", arguments={}, version="2.0.0")

        # Test v1.0.0
        result_v1 = evaluate([test_case], [output], [check_v1])
        check_result_v1 = result_v1.results[0].check_results[0]

        assert check_result_v1.status == "completed"
        assert check_result_v1.results["version"] == "1.0.0"

        # Test v2.0.0
        result_v2 = evaluate([test_case], [output], [check_v2])
        check_result_v2 = result_v2.results[0].check_results[0]

        assert check_result_v2.status == "completed"
        assert check_result_v2.results["version"] == "2.0.0"

    def test_engine_uses_latest_version_when_none_specified(self):
        """Test that engine uses latest version when Check.version is None."""

        @register("latest_test", version="1.0.0")
        class LatestTest_v1(BaseCheck):  # noqa: N801
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True, "version": "1.0.0"}

        @register("latest_test", version="2.1.0")
        class LatestTest_v2_1(BaseCheck):  # noqa: N801
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True, "version": "2.1.0"}

        @register("latest_test", version="2.0.0")
        class LatestTest_v2_0(BaseCheck):  # noqa: N801
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True, "version": "2.0.0"}

        # Create test case and output
        test_case = TestCase(id="test1", input="test input")
        output = Output(value="test output")

        # Create check without version (should use latest)
        check_latest = Check(type="latest_test", arguments={}, version=None)

        result = evaluate([test_case], [output], [check_latest])
        check_result = result.results[0].check_results[0]

        assert check_result.status == "completed"
        assert check_result.results["version"] == "2.1.0"  # Should be latest

    def test_engine_async_check_versioning(self):
        """Test that engine handles async check versioning correctly."""

        @register("async_version_test", version="1.0.0")
        class AsyncVersionTest_v1(BaseAsyncCheck):  # noqa: N801
            async def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True, "version": "1.0.0"}

        @register("async_version_test", version="2.0.0")
        class AsyncVersionTest_v2(BaseAsyncCheck):  # noqa: N801
            async def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True, "version": "2.0.0"}

        # Create test case and output
        test_case = TestCase(id="test1", input="test input")
        output = Output(value="test output")

        # Create checks with specific versions
        check_v1 = Check(type="async_version_test", arguments={}, version="1.0.0")
        check_v2 = Check(type="async_version_test", arguments={}, version="2.0.0")

        # Test v1.0.0
        result_v1 = evaluate([test_case], [output], [check_v1])
        check_result_v1 = result_v1.results[0].check_results[0]

        assert check_result_v1.status == "completed"
        assert check_result_v1.results["version"] == "1.0.0"

        # Test v2.0.0
        result_v2 = evaluate([test_case], [output], [check_v2])
        check_result_v2 = result_v2.results[0].check_results[0]

        assert check_result_v2.status == "completed"
        assert check_result_v2.results["version"] == "2.0.0"

    def test_engine_mixed_sync_async_versions(self):
        """Test engine with mixed sync/async checks across versions."""

        @register("mixed_test", version="1.0.0")
        class MixedTest_v1_sync(BaseCheck):  # noqa: N801
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True, "type": "sync", "version": "1.0.0"}

        @register("mixed_test", version="2.0.0")
        class MixedTest_v2_async(BaseAsyncCheck):  # noqa: N801
            async def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True, "type": "async", "version": "2.0.0"}

        # Create test case and output
        test_case = TestCase(id="test1", input="test input")
        output = Output(value="test output")

        # Create checks with different versions (sync and async)
        check_sync = Check(type="mixed_test", arguments={}, version="1.0.0")
        check_async = Check(type="mixed_test", arguments={}, version="2.0.0")

        # Test both in same evaluation
        result = evaluate([test_case], [output], [check_sync, check_async])

        assert len(result.results[0].check_results) == 2

        # Find sync and async results
        sync_result = None
        async_result = None

        for check_result in result.results[0].check_results:
            if check_result.results["type"] == "sync":
                sync_result = check_result
            elif check_result.results["type"] == "async":
                async_result = check_result

        assert sync_result is not None
        assert async_result is not None

        assert sync_result.results["version"] == "1.0.0"
        assert async_result.results["version"] == "2.0.0"

        assert sync_result.status == "completed"
        assert async_result.status == "completed"

    def test_engine_version_not_found_error(self):
        """Test engine handles gracefully when requested version doesn't exist."""

        @register("error_test", version="1.0.0")
        class ErrorTest_v1(BaseCheck):  # noqa: N801
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True, "version": "1.0.0"}

        # Create test case and output
        test_case = TestCase(id="test1", input="test input")
        output = Output(value="test output")

        # Create check with non-existent version
        check_nonexistent = Check(type="error_test", arguments={}, version="2.0.0")

        result = evaluate([test_case], [output], [check_nonexistent])
        check_result = result.results[0].check_results[0]

        # Should result in an error status
        assert check_result.status == "error"
        assert "Version '2.0.0' not found" in check_result.error.message

    def test_engine_version_in_check_result_metadata(self):
        """Test that check version is included in CheckResult metadata."""

        @register("metadata_test", version="1.5.0")
        class MetadataTest(BaseCheck):
            def __call__(self, **kwargs: Any) -> dict[str, Any]:  # noqa
                return {"passed": True}

        # Create test case and output
        test_case = TestCase(id="test1", input="test input")
        output = Output(value="test output")

        # Create check with specific version
        check = Check(type="metadata_test", arguments={}, version="1.5.0")

        result = evaluate([test_case], [output], [check])
        check_result = result.results[0].check_results[0]

        assert check_result.status == "completed"
        # The version should be included in metadata by the check execution
        assert check_result.metadata is not None
        assert "check_version" in check_result.metadata
        assert check_result.metadata["check_version"] == "1.5.0"
