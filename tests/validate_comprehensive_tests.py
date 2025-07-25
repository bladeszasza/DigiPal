#!/usr/bin/env python3
"""
Comprehensive test validation script for DigiPal system.

This script validates that all comprehensive test components are working
correctly and provides a summary of test coverage and quality.
"""

import sys
import subprocess
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_test_validation():
    """Run comprehensive test validation."""
    print("🚀 DigiPal Comprehensive Test Validation")
    print("=" * 50)
    
    validation_results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'tests': {},
        'summary': {}
    }
    
    # Test categories to validate
    test_categories = [
        {
            'name': 'Data Generation Tests',
            'command': ['python', '-m', 'pytest', 'tests/test_data_generation.py', '-v'],
            'timeout': 60
        },
        {
            'name': 'Coverage Analysis Tests',
            'command': ['python', '-m', 'pytest', 'tests/test_coverage_analysis.py', '-v'],
            'timeout': 30
        },
        {
            'name': 'Quality Analysis Tests',
            'command': ['python', '-m', 'pytest', 'tests/test_quality_analysis.py', '-v'],
            'timeout': 30
        },
        {
            'name': 'Requirements Compliance Tests',
            'command': ['python', '-m', 'pytest', 'tests/test_requirements_compliance.py', '-v'],
            'timeout': 120
        },
        {
            'name': 'Performance Benchmarks (Sample)',
            'command': ['python', '-m', 'pytest', 'tests/test_performance_benchmarks.py::TestPerformanceBenchmarks::test_pet_creation_performance', '-v'],
            'timeout': 60
        },
        {
            'name': 'End-to-End Lifecycle (Sample)',
            'command': ['python', '-m', 'pytest', 'tests/test_end_to_end_lifecycle.py::TestCompleteLifecycleScenarios::test_complete_new_user_journey', '-v'],
            'timeout': 120
        }
    ]
    
    successful_tests = 0
    total_tests = len(test_categories)
    
    for test_category in test_categories:
        print(f"\n🧪 Running {test_category['name']}...")
        
        try:
            start_time = time.time()
            result = subprocess.run(
                test_category['command'],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=test_category['timeout']
            )
            end_time = time.time()
            
            duration = end_time - start_time
            success = result.returncode == 0
            
            if success:
                successful_tests += 1
                print(f"  ✅ PASSED ({duration:.1f}s)")
            else:
                print(f"  ❌ FAILED ({duration:.1f}s)")
                print(f"  Error output: {result.stderr[:200]}...")
            
            validation_results['tests'][test_category['name']] = {
                'success': success,
                'duration': duration,
                'returncode': result.returncode,
                'stdout_lines': len(result.stdout.split('\n')),
                'stderr_lines': len(result.stderr.split('\n'))
            }
            
        except subprocess.TimeoutExpired:
            print(f"  ⏰ TIMEOUT (>{test_category['timeout']}s)")
            validation_results['tests'][test_category['name']] = {
                'success': False,
                'duration': test_category['timeout'],
                'error': 'timeout'
            }
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            validation_results['tests'][test_category['name']] = {
                'success': False,
                'error': str(e)
            }
    
    # Generate summary
    success_rate = (successful_tests / total_tests) * 100
    validation_results['summary'] = {
        'total_tests': total_tests,
        'successful_tests': successful_tests,
        'success_rate': success_rate,
        'overall_status': 'PASS' if success_rate >= 80 else 'FAIL'
    }
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    print(f"Total test categories: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Overall status: {validation_results['summary']['overall_status']}")
    
    # Detailed results
    print(f"\nDETAILED RESULTS:")
    for test_name, result in validation_results['tests'].items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        duration = result.get('duration', 0)
        print(f"  {status} {test_name} ({duration:.1f}s)")
    
    print("\n" + "=" * 50)
    
    return validation_results


def run_comprehensive_test_runner():
    """Test the comprehensive test runner itself."""
    print("\n🔧 Testing Comprehensive Test Runner...")
    
    try:
        # Test data generation suite
        result = subprocess.run(
            ['python', 'tests/run_comprehensive_tests.py', '--suite=data'],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print("  ✅ Comprehensive test runner working")
            return True
        else:
            print("  ❌ Comprehensive test runner failed")
            print(f"  Error: {result.stderr[:200]}...")
            return False
            
    except Exception as e:
        print(f"  ❌ Error testing comprehensive runner: {e}")
        return False


def run_analysis_tools():
    """Test the analysis tools."""
    print("\n🔍 Testing Analysis Tools...")
    
    tools_working = 0
    total_tools = 3
    
    # Test coverage analysis
    try:
        from tests.test_coverage_analysis import run_coverage_analysis
        coverage_report = run_coverage_analysis()
        if coverage_report and 'summary' in coverage_report:
            print("  ✅ Coverage analysis working")
            tools_working += 1
        else:
            print("  ❌ Coverage analysis failed")
    except Exception as e:
        print(f"  ❌ Coverage analysis error: {e}")
    
    # Test quality analysis
    try:
        from tests.test_quality_analysis import run_quality_analysis
        quality_report = run_quality_analysis()
        if quality_report and 'summary' in quality_report:
            print("  ✅ Quality analysis working")
            tools_working += 1
        else:
            print("  ❌ Quality analysis failed")
    except Exception as e:
        print(f"  ❌ Quality analysis error: {e}")
    
    # Test requirements compliance
    try:
        from tests.test_requirements_compliance import run_compliance_validation
        compliance_report = run_compliance_validation()
        if compliance_report and 'summary' in compliance_report:
            print("  ✅ Requirements compliance validation working")
            tools_working += 1
        else:
            print("  ❌ Requirements compliance validation failed")
    except Exception as e:
        print(f"  ❌ Requirements compliance error: {e}")
    
    success_rate = (tools_working / total_tools) * 100
    print(f"  Analysis tools success rate: {success_rate:.1f}%")
    
    return tools_working == total_tools


def main():
    """Main validation function."""
    print("Starting DigiPal Comprehensive Test Validation...")
    
    # Run test validation
    validation_results = run_test_validation()
    
    # Test comprehensive runner
    runner_working = run_comprehensive_test_runner()
    
    # Test analysis tools
    tools_working = run_analysis_tools()
    
    # Final summary
    print("\n" + "🏁 FINAL VALIDATION SUMMARY")
    print("=" * 50)
    
    test_success = validation_results['summary']['success_rate'] >= 80
    
    print(f"Test Categories: {'✅ PASS' if test_success else '❌ FAIL'} ({validation_results['summary']['success_rate']:.1f}%)")
    print(f"Test Runner: {'✅ PASS' if runner_working else '❌ FAIL'}")
    print(f"Analysis Tools: {'✅ PASS' if tools_working else '❌ FAIL'}")
    
    overall_success = test_success and runner_working and tools_working
    print(f"\nOverall Status: {'✅ COMPREHENSIVE TEST SUITE READY' if overall_success else '❌ ISSUES DETECTED'}")
    
    if overall_success:
        print("\n🎉 The comprehensive test suite is ready for use!")
        print("   You can now run: python tests/run_comprehensive_tests.py")
    else:
        print("\n⚠️  Some issues were detected. Please review the output above.")
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())