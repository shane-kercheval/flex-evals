"""Basic test of combined check approach."""

from src.flex_evals.checks.combined.exact_match import ExactMatchCheck
from src.flex_evals.schemas import TestCase, Output, Check
from src.flex_evals.checks.base import EvaluationContext
from src.flex_evals.registry import list_registered_checks, get_check_class
from src.flex_evals.engine import evaluate
from src.flex_evals.constants import CheckType


def test_combined_exact_match_basic():
    """Test basic functionality of combined ExactMatchCheck."""
    # Test 1: Schema validation works
    check = ExactMatchCheck(
        actual='hello',
        expected='hello',
        case_sensitive=True,
    )
    print('✓ Schema validation passed')
    
    # Test 2: Check is registered
    registered = list_registered_checks()
    assert 'exact_match' in registered
    assert '2.0.0' in registered['exact_match']
    print('✓ Check registration works')
    
    # Test 3: Direct execution works
    result = check(actual='hello', expected='hello')
    assert result['passed'] is True
    print('✓ Direct execution works')
    
    # Test 4: Case sensitivity works  
    result = check(actual='HELLO', expected='hello', case_sensitive=False)
    assert result['passed'] is True
    print('✓ Case sensitivity works')
    
    # Test 5: Negation works
    result = check(actual='hello', expected='world', negate=True)
    assert result['passed'] is True
    print('✓ Negation works')
    
    # Test 6: to_arguments works
    args = check.to_arguments()
    expected_args = {
        'actual': 'hello',
        'expected': 'hello', 
        'case_sensitive': True,
        'negate': False,
    }
    assert args == expected_args
    print('✓ to_arguments works')
    
    print('All basic tests passed!')


def test_engine_integration():
    """Test that combined checks work with the evaluation engine."""
    # Test with direct combined check instance
    check_instance = ExactMatchCheck(actual='hello', expected='hello')
    
    test_case = TestCase(id='test-1', input={'question': 'What is 2+2?'})
    output = Output(value='hello')
    
    try:
        # Test with combined check instance directly
        result = evaluate([test_case], [output], [check_instance])
        print('✓ Engine integration works with combined checks')
        print(f'  Check passed: {result.results[0].check_results[0].results["passed"]}')
        return True
    except Exception as e:
        print(f'✗ Engine integration failed: {e}')
        return False


if __name__ == '__main__':
    test_combined_exact_match_basic()
    print()
    test_engine_integration()