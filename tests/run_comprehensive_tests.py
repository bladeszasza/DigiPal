#!/usr/bin/env python3
"""
Comprehensive test runner for DigiPal system.

This script runs all test suites and generates comprehensive reports
including coverage, performance metrics, and validation results.
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class ComprehensiveTestRunner:
    """Runs comprehensive test suite with reporting."""
    
    def __init__(self, output_dir="test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_suites': {},
            'coverage': {},
            'performance': {},
            'summary': {}
        }
    
    def run_unit_tests(self):
        """Run all unit tests."""
        print("🧪 Running Unit Tests...")
        
        unit_test_files = [
            'test_digipal_core.py',
            'test_attribute_engine.py',
            'test_evolution_controller.py',
            'test_storage.py',
            'test_models.py',
            'test_ai_communication.py',
            'test_language_model.py',
            'test_speech_processor.py',
            'test_image_generator.py',
            'test_auth_manager.py',
            'test_auth_models.py',
            'test_session_manager.py',
            'test_gradio_interface.py',
            'test_mcp_server.py',
            'test_error_handling.py'
        ]
        
        unit_results = {}
        
        for test_file in unit_test_files:
            test_path = project_root / 'tests' / test_file
            if test_path.exists():
                print(f"  Running {test_file}...")
                result = self._run_pytest(test_path)
                unit_results[test_file] = result
            else:
                print(f"  ⚠️  {test_file} not found, skipping...")
                unit_results[test_file] = {'status': 'skipped', 'reason': 'file not found'}
        
        self.results['test_suites']['unit_tests'] = unit_results
        return unit_results
    
    def run_integration_tests(self):
        """Run integration tests."""
        print("🔗 Running Integration Tests...")
        
        integration_test_files = [
            'test_integration_main_flow.py',
            'test_auth_integration.py',
            'test_qwen_integration.py'
        ]
        
        integration_results = {}
        
        for test_file in integration_test_files:
            test_path = project_root / 'tests' / test_file
            if test_path.exists():
                print(f"  Running {test_file}...")
                result = self._run_pytest(test_path)
                integration_results[test_file] = result
            else:
                print(f"  ⚠️  {test_file} not found, skipping...")
                integration_results[test_file] = {'status': 'skipped', 'reason': 'file not found'}
        
        self.results['test_suites']['integration_tests'] = integration_results
        return integration_results
    
    def run_end_to_end_tests(self):
        """Run end-to-end tests."""
        print("🎯 Running End-to-End Tests...")
        
        e2e_test_files = [
            'test_end_to_end_lifecycle.py'
        ]
        
        e2e_results = {}
        
        for test_file in e2e_test_files:
            test_path = project_root / 'tests' / test_file
            if test_path.exists():
                print(f"  Running {test_file}...")
                result = self._run_pytest(test_path)
                e2e_results[test_file] = result
            else:
                print(f"  ⚠️  {test_file} not found, skipping...")
                e2e_results[test_file] = {'status': 'skipped', 'reason': 'file not found'}
        
        self.results['test_suites']['end_to_end_tests'] = e2e_results
        return e2e_results
    
    def run_performance_tests(self):
        """Run performance and load tests."""
        print("⚡ Running Performance Tests...")
        
        performance_test_files = [
            'test_performance_benchmarks.py',
            'test_memory_performance.py'
        ]
        
        performance_results = {}
        
        for test_file in performance_test_files:
            test_path = project_root / 'tests' / test_file
            if test_path.exists():
                print(f"  Running {test_file}...")
                result = self._run_pytest(test_path)
                performance_results[test_file] = result
            else:
                print(f"  ⚠️  {test_file} not found, skipping...")
                performance_results[test_file] = {'status': 'skipped', 'reason': 'file not found'}
        
        self.results['test_suites']['performance_tests'] = performance_results
        return performance_results
    
    def run_validation_tests(self):
        """Run comprehensive validation tests."""
        print("✅ Running Validation Tests...")
        
        validation_test_files = [
            'test_comprehensive_validation.py',
            'test_requirements_compliance.py'
        ]
        
        validation_results = {}
        
        for test_file in validation_test_files:
            test_path = project_root / 'tests' / test_file
            if test_path.exists():
                print(f"  Running {test_file}...")
                result = self._run_pytest(test_path)
                validation_results[test_file] = result
            else:
                print(f"  ⚠️  {test_file} not found, skipping...")
                validation_results[test_file] = {'status': 'skipped', 'reason': 'file not found'}
        
        self.results['test_suites']['validation_tests'] = validation_results
        return validation_results
    
    def run_quality_analysis(self):
        """Run test quality analysis."""
        print("🔍 Running Test Quality Analysis...")
        
        try:
            from tests.test_quality_analysis import run_quality_analysis
            quality_report = run_quality_analysis()
            
            quality_result = {
                'status': 'success',
                'report': quality_report
            }
            
            print(f"  ✅ Quality analysis completed")
            
        except Exception as e:
            print(f"  ❌ Quality analysis failed: {e}")
            quality_result = {
                'status': 'error',
                'error': str(e)
            }
        
        self.results['quality_analysis'] = quality_result
        return quality_result
    
    def run_requirements_compliance(self):
        """Run requirements compliance validation."""
        print("📋 Running Requirements Compliance Validation...")
        
        try:
            from tests.test_requirements_compliance import run_compliance_validation
            compliance_report = run_compliance_validation()
            
            compliance_result = {
                'status': 'success',
                'report': compliance_report
            }
            
            print(f"  ✅ Requirements compliance validation completed")
            
        except Exception as e:
            print(f"  ❌ Requirements compliance validation failed: {e}")
            compliance_result = {
                'status': 'error',
                'error': str(e)
            }
        
        self.results['requirements_compliance'] = compliance_result
        return compliance_result
    
    def run_data_generation_tests(self):
        """Run data generation tests."""
        print("🎲 Running Data Generation Tests...")
        
        data_test_files = [
            'test_data_generation.py'
        ]
        
        data_results = {}
        
        for test_file in data_test_files:
            test_path = project_root / 'tests' / test_file
            if test_path.exists():
                print(f"  Running {test_file}...")
                result = self._run_pytest(test_path)
                data_results[test_file] = result
            else:
                print(f"  ⚠️  {test_file} not found, skipping...")
                data_results[test_file] = {'status': 'skipped', 'reason': 'file not found'}
        
        self.results['test_suites']['data_generation_tests'] = data_results
        return data_results
    
    def run_coverage_analysis(self):
        """Run code coverage analysis."""
        print("📊 Running Coverage Analysis...")
        
        try:
            # Run pytest with coverage
            cmd = [
                'python', '-m', 'pytest',
                '--cov=digipal',
                '--cov-report=html:' + str(self.output_dir / 'coverage_html'),
                '--cov-report=json:' + str(self.output_dir / 'coverage.json'),
                '--cov-report=term-missing',
                '--cov-fail-under=0',  # Don't fail on low coverage
                'tests/'
            ]
            
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            coverage_result = {
                'status': 'success' if result.returncode == 0 else 'failed',
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            # Try to load coverage JSON if it exists
            coverage_json_path = self.output_dir / 'coverage.json'
            if coverage_json_path.exists():
                with open(coverage_json_path, 'r') as f:
                    coverage_data = json.load(f)
                    coverage_result['coverage_data'] = coverage_data
                    coverage_result['total_coverage'] = coverage_data.get('totals', {}).get('percent_covered', 0)
            
            # Run custom coverage analysis
            try:
                from tests.test_coverage_analysis import run_coverage_analysis
                custom_coverage = run_coverage_analysis()
                coverage_result['custom_analysis'] = custom_coverage
            except Exception as e:
                print(f"  ⚠️  Custom coverage analysis failed: {e}")
            
            self.results['coverage'] = coverage_result
            
            if coverage_result['status'] == 'success':
                print(f"  ✅ Coverage analysis completed")
                if 'total_coverage' in coverage_result:
                    print(f"  📈 Total coverage: {coverage_result['total_coverage']:.1f}%")
            else:
                print(f"  ❌ Coverage analysis failed")
            
            return coverage_result
            
        except subprocess.TimeoutExpired:
            print("  ⏰ Coverage analysis timed out")
            return {'status': 'timeout', 'error': 'Coverage analysis timed out'}
        except Exception as e:
            print(f"  ❌ Coverage analysis error: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _run_pytest(self, test_path, timeout=120):
        """Run pytest on a specific test file."""
        try:
            cmd = [
                'python', '-m', 'pytest',
                str(test_path),
                '-v',
                '--tb=short'
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            end_time = time.time()
            
            # Parse test results from stdout
            test_details = self._parse_pytest_output(result.stdout)
            
            return {
                'status': 'success' if result.returncode == 0 else 'failed',
                'returncode': result.returncode,
                'duration': end_time - start_time,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'test_details': test_details
            }
            
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'error': f'Test timed out after {timeout} seconds'
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def generate_summary(self):
        """Generate test summary."""
        print("📋 Generating Test Summary...")
        
        summary = {
            'total_test_suites': len(self.results['test_suites']),
            'successful_suites': 0,
            'failed_suites': 0,
            'skipped_suites': 0,
            'total_duration': 0,
            'coverage_percentage': 0
        }
        
        # Analyze test suite results
        for suite_name, suite_results in self.results['test_suites'].items():
            suite_success = True
            suite_duration = 0
            
            for test_file, test_result in suite_results.items():
                if test_result.get('status') == 'failed':
                    suite_success = False
                elif test_result.get('status') == 'skipped':
                    continue
                
                suite_duration += test_result.get('duration', 0)
            
            if suite_success:
                summary['successful_suites'] += 1
            else:
                summary['failed_suites'] += 1
            
            summary['total_duration'] += suite_duration
        
        # Add coverage information
        if 'coverage' in self.results and 'total_coverage' in self.results['coverage']:
            summary['coverage_percentage'] = self.results['coverage']['total_coverage']
        
        self.results['summary'] = summary
        
        # Print summary
        print(f"\n📊 Test Summary:")
        print(f"  Total test suites: {summary['total_test_suites']}")
        print(f"  Successful: {summary['successful_suites']}")
        print(f"  Failed: {summary['failed_suites']}")
        print(f"  Skipped: {summary['skipped_suites']}")
        print(f"  Total duration: {summary['total_duration']:.2f}s")
        print(f"  Coverage: {summary['coverage_percentage']:.1f}%")
        
        return summary
    
    def save_results(self):
        """Save test results to file."""
        results_file = self.output_dir / 'comprehensive_test_results.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"💾 Results saved to {results_file}")
        
        # Also create a human-readable summary
        summary_file = self.output_dir / 'test_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("DigiPal Comprehensive Test Results\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Timestamp: {self.results['timestamp']}\n\n")
            
            # Summary
            summary = self.results['summary']
            f.write("Summary:\n")
            f.write(f"  Total test suites: {summary['total_test_suites']}\n")
            f.write(f"  Successful: {summary['successful_suites']}\n")
            f.write(f"  Failed: {summary['failed_suites']}\n")
            f.write(f"  Total duration: {summary['total_duration']:.2f}s\n")
            f.write(f"  Coverage: {summary['coverage_percentage']:.1f}%\n\n")
            
            # Detailed results
            for suite_name, suite_results in self.results['test_suites'].items():
                f.write(f"{suite_name.replace('_', ' ').title()}:\n")
                for test_file, test_result in suite_results.items():
                    status = test_result.get('status', 'unknown')
                    duration = test_result.get('duration', 0)
                    f.write(f"  {test_file}: {status} ({duration:.2f}s)\n")
                f.write("\n")
        
        print(f"📄 Summary saved to {summary_file}")
    
    def _parse_pytest_output(self, output):
        """Parse pytest output to extract test information."""
        lines = output.split('\n')
        test_details = {
            'tests_run': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': []
        }
        
        for line in lines:
            if '::' in line and ('PASSED' in line or 'FAILED' in line or 'SKIPPED' in line):
                test_details['tests_run'] += 1
                if 'PASSED' in line:
                    test_details['passed'] += 1
                elif 'FAILED' in line:
                    test_details['failed'] += 1
                elif 'SKIPPED' in line:
                    test_details['skipped'] += 1
            elif 'ERROR' in line or 'FAILED' in line:
                test_details['errors'].append(line.strip())
        
        return test_details
    
    def run_all_tests(self):
        """Run all test suites."""
        print("🚀 Starting Comprehensive Test Suite")
        print("=" * 50)
        
        start_time = time.time()
        
        # Run all test categories
        self.run_unit_tests()
        self.run_integration_tests()
        self.run_end_to_end_tests()
        self.run_performance_tests()
        self.run_validation_tests()
        self.run_data_generation_tests()
        
        # Run coverage analysis
        self.run_coverage_analysis()
        
        # Run quality analysis
        self.run_quality_analysis()
        
        # Run requirements compliance validation
        self.run_requirements_compliance()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        self.save_results()
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        print("\n" + "=" * 50)
        print(f"🏁 Comprehensive test suite completed in {total_duration:.2f}s")
        
        # Return overall success status
        summary = self.results['summary']
        return summary['failed_suites'] == 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run comprehensive DigiPal tests')
    parser.add_argument('--output-dir', default='test_results', 
                       help='Output directory for test results')
    parser.add_argument('--suite', choices=['unit', 'integration', 'e2e', 'performance', 'validation', 'data', 'all'],
                       default='all', help='Test suite to run')
    parser.add_argument('--coverage', action='store_true', 
                       help='Run coverage analysis')
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner(args.output_dir)
    
    if args.suite == 'all':
        success = runner.run_all_tests()
    else:
        # Run specific suite
        if args.suite == 'unit':
            runner.run_unit_tests()
        elif args.suite == 'integration':
            runner.run_integration_tests()
        elif args.suite == 'e2e':
            runner.run_end_to_end_tests()
        elif args.suite == 'performance':
            runner.run_performance_tests()
        elif args.suite == 'validation':
            runner.run_validation_tests()
        elif args.suite == 'data':
            runner.run_data_generation_tests()
        
        if args.coverage:
            runner.run_coverage_analysis()
        
        runner.generate_summary()
        runner.save_results()
        
        success = runner.results['summary']['failed_suites'] == 0
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()