"""
Test quality analysis for DigiPal system.

This module analyzes the quality and comprehensiveness of tests,
identifying areas where test coverage could be improved.
"""

import pytest
import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Any
from collections import defaultdict


class TestQualityAnalyzer:
    """Analyzes the quality and comprehensiveness of test suites."""
    
    def __init__(self, test_dir: str = "tests"):
        self.test_dir = Path(test_dir)
        self.quality_metrics = {
            'test_files': {},
            'test_patterns': {},
            'assertion_analysis': {},
            'fixture_usage': {},
            'mock_usage': {},
            'edge_case_coverage': {},
            'summary': {}
        }
    
    def analyze_test_file(self, test_file: Path) -> Dict[str, Any]:
        """Analyze a single test file for quality metrics."""
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
            
            analysis = {
                'file': str(test_file),
                'test_classes': 0,
                'test_methods': 0,
                'assertions': 0,
                'fixtures': 0,
                'mocks': 0,
                'parametrized_tests': 0,
                'async_tests': 0,
                'edge_case_tests': 0,
                'integration_tests': 0,
                'assertion_types': defaultdict(int),
                'test_patterns': [],
                'complexity_score': 0
            }
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                    analysis['test_classes'] += 1
                    
                    # Check for integration test patterns
                    if 'integration' in node.name.lower() or 'end_to_end' in node.name.lower():
                        analysis['integration_tests'] += 1
                
                elif isinstance(node, ast.FunctionDef):
                    if node.name.startswith('test_'):
                        analysis['test_methods'] += 1
                        
                        # Check for async tests
                        if isinstance(node, ast.AsyncFunctionDef):
                            analysis['async_tests'] += 1
                        
                        # Check for parametrized tests
                        for decorator in node.decorator_list:
                            if (isinstance(decorator, ast.Call) and 
                                isinstance(decorator.func, ast.Attribute) and
                                decorator.func.attr == 'parametrize'):
                                analysis['parametrized_tests'] += 1
                        
                        # Check for edge case patterns
                        if any(keyword in node.name.lower() for keyword in 
                               ['edge', 'boundary', 'limit', 'error', 'exception', 'invalid']):
                            analysis['edge_case_tests'] += 1
                    
                    elif node.name.startswith('fixture') or any(
                        isinstance(d, ast.Name) and d.id == 'fixture' for d in node.decorator_list):
                        analysis['fixtures'] += 1
                
                elif isinstance(node, ast.Call):
                    # Count assertions
                    if (isinstance(node.func, ast.Name) and 
                        node.func.id.startswith('assert')):
                        analysis['assertions'] += 1
                        analysis['assertion_types'][node.func.id] += 1
                    
                    # Count mocks
                    elif (isinstance(node.func, ast.Attribute) and
                          'mock' in str(node.func).lower()):
                        analysis['mocks'] += 1
                    
                    # Check for Mock/patch usage
                    elif (isinstance(node.func, ast.Name) and
                          node.func.id in ['Mock', 'MagicMock', 'patch']):
                        analysis['mocks'] += 1
            
            # Calculate complexity score
            analysis['complexity_score'] = self._calculate_complexity_score(analysis)
            
            # Identify test patterns
            analysis['test_patterns'] = self._identify_test_patterns(content)
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing {test_file}: {e}")
            return {'file': str(test_file), 'error': str(e)}
    
    def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate a complexity score for the test file."""
        score = 0.0
        
        # Base score from test count
        score += analysis['test_methods'] * 1.0
        
        # Bonus for different test types
        score += analysis['parametrized_tests'] * 2.0  # Parametrized tests are more valuable
        score += analysis['async_tests'] * 1.5         # Async tests are more complex
        score += analysis['edge_case_tests'] * 2.0     # Edge cases are important
        score += analysis['integration_tests'] * 3.0   # Integration tests are most valuable
        
        # Bonus for good assertion coverage
        if analysis['test_methods'] > 0:
            assertions_per_test = analysis['assertions'] / analysis['test_methods']
            if assertions_per_test >= 2.0:  # Good assertion coverage
                score += analysis['test_methods'] * 0.5
        
        # Bonus for fixture usage (indicates good test structure)
        score += analysis['fixtures'] * 1.0
        
        # Bonus for mock usage (indicates isolation testing)
        score += min(analysis['mocks'], analysis['test_methods']) * 0.5
        
        return score
    
    def _identify_test_patterns(self, content: str) -> List[str]:
        """Identify common test patterns in the content."""
        patterns = []
        
        # Test pattern indicators
        pattern_indicators = {
            'setup_teardown': r'def (setup|teardown|setUp|tearDown)',
            'fixture_usage': r'@pytest\.fixture',
            'parametrized': r'@pytest\.mark\.parametrize',
            'mock_usage': r'(Mock|patch|mock)',
            'exception_testing': r'pytest\.raises|assertRaises',
            'async_testing': r'async def test_|@pytest\.mark\.asyncio',
            'integration_testing': r'class.*Integration.*Test|def test.*integration',
            'performance_testing': r'def test.*performance|def test.*benchmark',
            'edge_case_testing': r'def test.*(edge|boundary|limit|invalid)',
            'happy_path_testing': r'def test.*(success|valid|normal|typical)'
        }
        
        for pattern_name, regex in pattern_indicators.items():
            if re.search(regex, content, re.IGNORECASE):
                patterns.append(pattern_name)
        
        return patterns
    
    def analyze_all_tests(self) -> Dict[str, Any]:
        """Analyze all test files and generate comprehensive quality report."""
        all_analyses = []
        
        # Analyze each test file
        for test_file in self.test_dir.rglob("test_*.py"):
            analysis = self.analyze_test_file(test_file)
            all_analyses.append(analysis)
        
        # Generate summary statistics
        summary = self._generate_summary(all_analyses)
        
        # Identify quality issues
        quality_issues = self._identify_quality_issues(all_analyses)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_analyses, quality_issues)
        
        return {
            'file_analyses': all_analyses,
            'summary': summary,
            'quality_issues': quality_issues,
            'recommendations': recommendations
        }
    
    def _generate_summary(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from all analyses."""
        total_files = len([a for a in analyses if 'error' not in a])
        total_test_classes = sum(a.get('test_classes', 0) for a in analyses)
        total_test_methods = sum(a.get('test_methods', 0) for a in analyses)
        total_assertions = sum(a.get('assertions', 0) for a in analyses)
        total_fixtures = sum(a.get('fixtures', 0) for a in analyses)
        total_mocks = sum(a.get('mocks', 0) for a in analyses)
        total_parametrized = sum(a.get('parametrized_tests', 0) for a in analyses)
        total_async = sum(a.get('async_tests', 0) for a in analyses)
        total_edge_cases = sum(a.get('edge_case_tests', 0) for a in analyses)
        total_integration = sum(a.get('integration_tests', 0) for a in analyses)
        
        avg_assertions_per_test = (total_assertions / total_test_methods) if total_test_methods > 0 else 0
        avg_complexity_score = sum(a.get('complexity_score', 0) for a in analyses) / total_files if total_files > 0 else 0
        
        return {
            'total_files': total_files,
            'total_test_classes': total_test_classes,
            'total_test_methods': total_test_methods,
            'total_assertions': total_assertions,
            'total_fixtures': total_fixtures,
            'total_mocks': total_mocks,
            'total_parametrized': total_parametrized,
            'total_async': total_async,
            'total_edge_cases': total_edge_cases,
            'total_integration': total_integration,
            'avg_assertions_per_test': avg_assertions_per_test,
            'avg_complexity_score': avg_complexity_score,
            'test_method_distribution': self._calculate_distribution(analyses, 'test_methods'),
            'assertion_distribution': self._calculate_distribution(analyses, 'assertions')
        }
    
    def _calculate_distribution(self, analyses: List[Dict[str, Any]], metric: str) -> Dict[str, int]:
        """Calculate distribution of a metric across test files."""
        values = [a.get(metric, 0) for a in analyses if 'error' not in a]
        
        distribution = {
            'min': min(values) if values else 0,
            'max': max(values) if values else 0,
            'avg': sum(values) / len(values) if values else 0,
            'files_with_zero': sum(1 for v in values if v == 0),
            'files_with_low': sum(1 for v in values if 0 < v <= 3),
            'files_with_medium': sum(1 for v in values if 3 < v <= 10),
            'files_with_high': sum(1 for v in values if v > 10)
        }
        
        return distribution
    
    def _identify_quality_issues(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify quality issues in the test suite."""
        issues = []
        
        for analysis in analyses:
            if 'error' in analysis:
                continue
            
            file_issues = []
            
            # Check for low assertion coverage
            if analysis['test_methods'] > 0:
                assertions_per_test = analysis['assertions'] / analysis['test_methods']
                if assertions_per_test < 1.5:
                    file_issues.append({
                        'type': 'low_assertion_coverage',
                        'severity': 'medium',
                        'message': f"Low assertion coverage: {assertions_per_test:.1f} assertions per test"
                    })
            
            # Check for no edge case tests
            if analysis['test_methods'] > 5 and analysis['edge_case_tests'] == 0:
                file_issues.append({
                    'type': 'no_edge_case_tests',
                    'severity': 'medium',
                    'message': "No edge case tests found"
                })
            
            # Check for no mock usage in large test files
            if analysis['test_methods'] > 10 and analysis['mocks'] == 0:
                file_issues.append({
                    'type': 'no_mock_usage',
                    'severity': 'low',
                    'message': "No mock usage found in large test file"
                })
            
            # Check for no fixtures in complex test files
            if analysis['test_methods'] > 8 and analysis['fixtures'] == 0:
                file_issues.append({
                    'type': 'no_fixtures',
                    'severity': 'low',
                    'message': "No fixtures found in complex test file"
                })
            
            # Check for very low complexity score
            if analysis['complexity_score'] < analysis['test_methods'] * 0.5:
                file_issues.append({
                    'type': 'low_complexity',
                    'severity': 'medium',
                    'message': f"Low test complexity score: {analysis['complexity_score']:.1f}"
                })
            
            if file_issues:
                issues.append({
                    'file': analysis['file'],
                    'issues': file_issues
                })
        
        return issues
    
    def _generate_recommendations(self, analyses: List[Dict[str, Any]], issues: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for improving test quality."""
        recommendations = []
        
        summary = self._generate_summary(analyses)
        
        # Recommendations based on summary statistics
        if summary['avg_assertions_per_test'] < 2.0:
            recommendations.append(
                "Increase assertion coverage: Add more assertions per test method to improve validation"
            )
        
        if summary['total_edge_cases'] < summary['total_test_methods'] * 0.2:
            recommendations.append(
                "Add more edge case tests: Include boundary conditions, error cases, and invalid inputs"
            )
        
        if summary['total_parametrized'] < summary['total_test_methods'] * 0.1:
            recommendations.append(
                "Use parametrized tests: Reduce code duplication by parametrizing similar test cases"
            )
        
        if summary['total_integration'] < 3:
            recommendations.append(
                "Add integration tests: Include tests that verify component interactions"
            )
        
        if summary['total_async'] == 0:
            recommendations.append(
                "Add async tests: Include tests for asynchronous functionality if applicable"
            )
        
        # Recommendations based on issues
        issue_types = set()
        for issue_group in issues:
            for issue in issue_group['issues']:
                issue_types.add(issue['type'])
        
        if 'low_assertion_coverage' in issue_types:
            recommendations.append(
                "Improve assertion coverage in specific files: Focus on files with low assertions per test"
            )
        
        if 'no_edge_case_tests' in issue_types:
            recommendations.append(
                "Add edge case tests to large test files: Include error conditions and boundary tests"
            )
        
        if 'no_mock_usage' in issue_types:
            recommendations.append(
                "Use mocks for isolation: Add mocks to isolate units under test from dependencies"
            )
        
        return recommendations
    
    def print_quality_report(self):
        """Print a formatted test quality report."""
        report = self.analyze_all_tests()
        
        print("=" * 60)
        print("DIGIPAL TEST QUALITY ANALYSIS")
        print("=" * 60)
        
        summary = report['summary']
        print(f"\nSUMMARY:")
        print(f"  Total test files: {summary['total_files']}")
        print(f"  Total test classes: {summary['total_test_classes']}")
        print(f"  Total test methods: {summary['total_test_methods']}")
        print(f"  Total assertions: {summary['total_assertions']}")
        print(f"  Average assertions per test: {summary['avg_assertions_per_test']:.1f}")
        print(f"  Average complexity score: {summary['avg_complexity_score']:.1f}")
        
        print(f"\nTEST TYPE DISTRIBUTION:")
        print(f"  Parametrized tests: {summary['total_parametrized']}")
        print(f"  Async tests: {summary['total_async']}")
        print(f"  Edge case tests: {summary['total_edge_cases']}")
        print(f"  Integration tests: {summary['total_integration']}")
        print(f"  Fixtures: {summary['total_fixtures']}")
        print(f"  Mocks: {summary['total_mocks']}")
        
        # Quality issues
        issues = report['quality_issues']
        if issues:
            print(f"\nQUALITY ISSUES ({len(issues)} files):")
            for issue_group in issues:
                file_name = Path(issue_group['file']).name
                print(f"  {file_name}:")
                for issue in issue_group['issues']:
                    severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                    icon = severity_icon.get(issue['severity'], "⚪")
                    print(f"    {icon} {issue['message']}")
        
        # Recommendations
        recommendations = report['recommendations']
        if recommendations:
            print(f"\nRECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"  {i}. {rec}")
        
        print("\n" + "=" * 60)
        
        return report


class TestQualityAnalysis:
    """Tests for the test quality analysis functionality."""
    
    def test_quality_analyzer_initialization(self):
        """Test TestQualityAnalyzer initialization."""
        analyzer = TestQualityAnalyzer()
        
        assert analyzer.test_dir == Path("tests")
        assert isinstance(analyzer.quality_metrics, dict)
    
    def test_test_file_analysis(self):
        """Test analysis of individual test files."""
        analyzer = TestQualityAnalyzer()
        
        # Find a test file to analyze
        test_files = list(analyzer.test_dir.rglob("test_*.py"))
        assert len(test_files) > 0, "No test files found"
        
        # Analyze first test file
        analysis = analyzer.analyze_test_file(test_files[0])
        
        assert 'file' in analysis
        assert 'test_methods' in analysis
        assert 'assertions' in analysis
        assert 'complexity_score' in analysis
        assert isinstance(analysis['test_patterns'], list)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis of all test files."""
        analyzer = TestQualityAnalyzer()
        report = analyzer.analyze_all_tests()
        
        assert 'file_analyses' in report
        assert 'summary' in report
        assert 'quality_issues' in report
        assert 'recommendations' in report
        
        summary = report['summary']
        assert 'total_files' in summary
        assert 'total_test_methods' in summary
        assert 'avg_assertions_per_test' in summary
        assert 'avg_complexity_score' in summary
    
    def test_quality_report_generation(self, capsys):
        """Test quality report printing."""
        analyzer = TestQualityAnalyzer()
        report = analyzer.print_quality_report()
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "DIGIPAL TEST QUALITY ANALYSIS" in output
        assert "SUMMARY:" in output
        assert "Total test files:" in output
        assert "Total test methods:" in output
        
        # Should return the report
        assert isinstance(report, dict)
        assert 'summary' in report


def run_quality_analysis():
    """Run complete test quality analysis and print report."""
    analyzer = TestQualityAnalyzer()
    return analyzer.print_quality_report()


if __name__ == "__main__":
    # Run quality analysis
    run_quality_analysis()
    
    # Run tests
    pytest.main([__file__, "-v"])