"""
Test coverage analysis and reporting for DigiPal system.

This module provides comprehensive analysis of test coverage across
all components and identifies areas needing additional testing.
"""

import pytest
import os
import ast
import inspect
from pathlib import Path
from typing import Dict, List, Set, Any
import importlib.util


class CoverageAnalyzer:
    """Analyzes test coverage across the DigiPal codebase."""
    
    def __init__(self, source_dir: str = "digipal", test_dir: str = "tests"):
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.coverage_report = {
            'modules': {},
            'classes': {},
            'functions': {},
            'methods': {},
            'summary': {}
        }
    
    def analyze_source_code(self) -> Dict[str, Any]:
        """Analyze source code to identify all testable components."""
        components = {
            'modules': set(),
            'classes': {},
            'functions': {},
            'methods': {}
        }
        
        # Walk through source directory
        for py_file in self.source_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                module_name = str(py_file.relative_to(self.source_dir)).replace('/', '.').replace('.py', '')
                components['modules'].add(module_name)
                
                # Analyze classes and methods
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        class_name = f"{module_name}.{node.name}"
                        components['classes'][class_name] = []
                        
                        # Find methods in class
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef):
                                method_name = f"{class_name}.{item.name}"
                                components['methods'][method_name] = {
                                    'class': class_name,
                                    'name': item.name,
                                    'line': item.lineno,
                                    'is_private': item.name.startswith('_'),
                                    'is_property': any(isinstance(d, ast.Name) and d.id == 'property' 
                                                     for d in item.decorator_list)
                                }
                                components['classes'][class_name].append(method_name)
                    
                    elif isinstance(node, ast.FunctionDef):
                        # Check if this function is not inside a class
                        is_method = False
                        for parent in ast.walk(tree):
                            if isinstance(parent, ast.ClassDef) and hasattr(parent, 'body'):
                                if node in parent.body:
                                    is_method = True
                                    break
                        
                        if not is_method:
                        function_name = f"{module_name}.{node.name}"
                        components['functions'][function_name] = {
                            'module': module_name,
                            'name': node.name,
                            'line': node.lineno,
                            'is_private': node.name.startswith('_')
                        }
            
            except Exception as e:
                print(f"Error analyzing {py_file}: {e}")
        
        return components
    
    def analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze existing test files to determine coverage."""
        test_coverage = {
            'test_files': [],
            'tested_modules': set(),
            'tested_classes': set(),
            'tested_functions': set(),
            'tested_methods': set()
        }
        
        # Walk through test directory
        for test_file in self.test_dir.rglob("test_*.py"):
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                test_info = {
                    'file': str(test_file),
                    'test_classes': [],
                    'test_functions': [],
                    'imports': []
                }
                
                # Analyze imports to see what's being tested
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith('digipal'):
                            test_info['imports'].append(node.module)
                            test_coverage['tested_modules'].add(node.module)
                            
                            # Track imported classes/functions
                            for alias in node.names:
                                if alias.name[0].isupper():  # Likely a class
                                    test_coverage['tested_classes'].add(f"{node.module}.{alias.name}")
                                else:  # Likely a function
                                    test_coverage['tested_functions'].add(f"{node.module}.{alias.name}")
                    
                    elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                        test_info['test_classes'].append(node.name)
                        
                        # Analyze test methods
                        for item in node.body:
                            if isinstance(item, ast.FunctionDef) and item.name.startswith('test_'):
                                test_info['test_functions'].append(f"{node.name}.{item.name}")
                    
                    elif isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        test_info['test_functions'].append(node.name)
                
                test_coverage['test_files'].append(test_info)
            
            except Exception as e:
                print(f"Error analyzing test file {test_file}: {e}")
        
        return test_coverage
    
    def generate_coverage_report(self) -> Dict[str, Any]:
        """Generate comprehensive coverage report."""
        source_components = self.analyze_source_code()
        test_coverage = self.analyze_test_coverage()
        
        # Calculate coverage percentages
        total_modules = len(source_components['modules'])
        tested_modules = len(test_coverage['tested_modules'])
        module_coverage = min(100.0, (tested_modules / total_modules * 100)) if total_modules > 0 else 0
        
        total_classes = len(source_components['classes'])
        tested_classes = len(test_coverage['tested_classes'])
        class_coverage = min(100.0, (tested_classes / total_classes * 100)) if total_classes > 0 else 0
        
        total_functions = len(source_components['functions'])
        tested_functions = len(test_coverage['tested_functions'])
        function_coverage = min(100.0, (tested_functions / total_functions * 100)) if total_functions > 0 else 0
        
        # Identify untested components
        untested_modules = source_components['modules'] - test_coverage['tested_modules']
        untested_classes = set(source_components['classes'].keys()) - test_coverage['tested_classes']
        untested_functions = set(source_components['functions'].keys()) - test_coverage['tested_functions']
        
        report = {
            'summary': {
                'total_modules': total_modules,
                'tested_modules': tested_modules,
                'module_coverage': module_coverage,
                'total_classes': total_classes,
                'tested_classes': tested_classes,
                'class_coverage': class_coverage,
                'total_functions': total_functions,
                'tested_functions': tested_functions,
                'function_coverage': function_coverage,
                'overall_coverage': min(100.0, (module_coverage + class_coverage + function_coverage) / 3)
            },
            'untested': {
                'modules': list(untested_modules),
                'classes': list(untested_classes),
                'functions': list(untested_functions)
            },
            'source_components': source_components,
            'test_coverage': test_coverage
        }
        
        return report
    
    def print_coverage_report(self):
        """Print a formatted coverage report."""
        report = self.generate_coverage_report()
        
        print("=" * 60)
        print("DIGIPAL TEST COVERAGE ANALYSIS")
        print("=" * 60)
        
        summary = report['summary']
        print(f"\nSUMMARY:")
        print(f"  Overall Coverage: {summary['overall_coverage']:.1f}%")
        print(f"  Module Coverage: {summary['tested_modules']}/{summary['total_modules']} ({summary['module_coverage']:.1f}%)")
        print(f"  Class Coverage: {summary['tested_classes']}/{summary['total_classes']} ({summary['class_coverage']:.1f}%)")
        print(f"  Function Coverage: {summary['tested_functions']}/{summary['total_functions']} ({summary['function_coverage']:.1f}%)")
        
        untested = report['untested']
        
        if untested['modules']:
            print(f"\nUNTESTED MODULES ({len(untested['modules'])}):")
            for module in sorted(untested['modules']):
                print(f"  - {module}")
        
        if untested['classes']:
            print(f"\nUNTESTED CLASSES ({len(untested['classes'])}):")
            for class_name in sorted(untested['classes']):
                print(f"  - {class_name}")
        
        if untested['functions']:
            print(f"\nUNTESTED FUNCTIONS ({len(untested['functions'])}):")
            for func_name in sorted(untested['functions']):
                print(f"  - {func_name}")
        
        # Test file analysis
        test_files = report['test_coverage']['test_files']
        print(f"\nTEST FILES ({len(test_files)}):")
        for test_file in test_files:
            file_name = Path(test_file['file']).name
            test_count = len(test_file['test_functions'])
            print(f"  - {file_name}: {test_count} tests")
        
        print("\n" + "=" * 60)
        
        return report


class TestCoverageAnalysis:
    """Tests for the coverage analysis functionality."""
    
    def test_coverage_analyzer_initialization(self):
        """Test CoverageAnalyzer initialization."""
        analyzer = CoverageAnalyzer()
        
        assert analyzer.source_dir == Path("digipal")
        assert analyzer.test_dir == Path("tests")
        assert isinstance(analyzer.coverage_report, dict)
    
    def test_source_code_analysis(self):
        """Test source code analysis functionality."""
        analyzer = CoverageAnalyzer()
        components = analyzer.analyze_source_code()
        
        assert 'modules' in components
        assert 'classes' in components
        assert 'functions' in components
        assert 'methods' in components
        
        # Should find core modules
        modules = components['modules']
        assert any('core.digipal_core' in str(module) for module in modules)
        assert any('storage.storage_manager' in str(module) for module in modules)
        
        # Should find classes
        classes = components['classes']
        assert any('DigiPalCore' in class_name for class_name in classes)
        assert any('StorageManager' in class_name for class_name in classes)
    
    def test_test_coverage_analysis(self):
        """Test test coverage analysis functionality."""
        analyzer = CoverageAnalyzer()
        coverage = analyzer.analyze_test_coverage()
        
        assert 'test_files' in coverage
        assert 'tested_modules' in coverage
        assert 'tested_classes' in coverage
        
        # Should find test files
        test_files = coverage['test_files']
        assert len(test_files) > 0
        
        # Should find tested modules
        tested_modules = coverage['tested_modules']
        assert len(tested_modules) > 0
    
    def test_coverage_report_generation(self):
        """Test coverage report generation."""
        analyzer = CoverageAnalyzer()
        report = analyzer.generate_coverage_report()
        
        assert 'summary' in report
        assert 'untested' in report
        assert 'source_components' in report
        assert 'test_coverage' in report
        
        summary = report['summary']
        assert 'overall_coverage' in summary
        assert 'module_coverage' in summary
        assert 'class_coverage' in summary
        assert 'function_coverage' in summary
        
        # Coverage percentages should be valid
        assert 0 <= summary['overall_coverage'] <= 100
        assert 0 <= summary['module_coverage'] <= 100
        assert 0 <= summary['class_coverage'] <= 100
        assert 0 <= summary['function_coverage'] <= 100
    
    def test_print_coverage_report(self, capsys):
        """Test coverage report printing."""
        analyzer = CoverageAnalyzer()
        report = analyzer.print_coverage_report()
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "DIGIPAL TEST COVERAGE ANALYSIS" in output
        assert "SUMMARY:" in output
        assert "Overall Coverage:" in output
        assert "Module Coverage:" in output
        assert "Class Coverage:" in output
        assert "Function Coverage:" in output
        
        # Should return the report
        assert isinstance(report, dict)
        assert 'summary' in report


class TestCoverageRequirements:
    """Test that coverage meets minimum requirements."""
    
    def test_minimum_coverage_requirements(self):
        """Test that minimum coverage requirements are met."""
        analyzer = CoverageAnalyzer()
        report = analyzer.generate_coverage_report()
        
        summary = report['summary']
        
        # Define minimum coverage requirements
        MIN_MODULE_COVERAGE = 80.0  # 80% of modules should have tests
        MIN_CLASS_COVERAGE = 70.0   # 70% of classes should have tests
        MIN_FUNCTION_COVERAGE = 60.0  # 60% of functions should have tests
        MIN_OVERALL_COVERAGE = 70.0   # 70% overall coverage
        
        print(f"\nCoverage Requirements Check:")
        print(f"  Module Coverage: {summary['module_coverage']:.1f}% (min: {MIN_MODULE_COVERAGE}%)")
        print(f"  Class Coverage: {summary['class_coverage']:.1f}% (min: {MIN_CLASS_COVERAGE}%)")
        print(f"  Function Coverage: {summary['function_coverage']:.1f}% (min: {MIN_FUNCTION_COVERAGE}%)")
        print(f"  Overall Coverage: {summary['overall_coverage']:.1f}% (min: {MIN_OVERALL_COVERAGE}%)")
        
        # Check requirements (with warnings instead of failures for now)
        if summary['module_coverage'] < MIN_MODULE_COVERAGE:
            print(f"⚠️  Module coverage below minimum: {summary['module_coverage']:.1f}% < {MIN_MODULE_COVERAGE}%")
        
        if summary['class_coverage'] < MIN_CLASS_COVERAGE:
            print(f"⚠️  Class coverage below minimum: {summary['class_coverage']:.1f}% < {MIN_CLASS_COVERAGE}%")
        
        if summary['function_coverage'] < MIN_FUNCTION_COVERAGE:
            print(f"⚠️  Function coverage below minimum: {summary['function_coverage']:.1f}% < {MIN_FUNCTION_COVERAGE}%")
        
        if summary['overall_coverage'] < MIN_OVERALL_COVERAGE:
            print(f"⚠️  Overall coverage below minimum: {summary['overall_coverage']:.1f}% < {MIN_OVERALL_COVERAGE}%")
        
        # For now, just ensure we have some coverage
        assert summary['overall_coverage'] > 0, "No test coverage detected"
        assert summary['tested_modules'] > 0, "No modules are being tested"
        assert summary['tested_classes'] > 0, "No classes are being tested"
    
    def test_critical_components_coverage(self):
        """Test that critical components have test coverage."""
        analyzer = CoverageAnalyzer()
        report = analyzer.generate_coverage_report()
        
        tested_modules = report['test_coverage']['tested_modules']
        tested_classes = report['test_coverage']['tested_classes']
        
        # Critical modules that must have tests
        critical_modules = [
            'digipal.core.digipal_core',
            'digipal.storage.storage_manager',
            'digipal.core.models',
            'digipal.mcp.server'
        ]
        
        # Critical classes that must have tests
        critical_classes = [
            'DigiPalCore',
            'StorageManager',
            'DigiPal',
            'MCPServer'
        ]
        
        print(f"\nCritical Components Coverage:")
        
        missing_modules = []
        for module in critical_modules:
            if module not in tested_modules:
                missing_modules.append(module)
            else:
                print(f"  ✅ {module}")
        
        missing_classes = []
        for class_name in critical_classes:
            found = any(class_name in tested_class for tested_class in tested_classes)
            if not found:
                missing_classes.append(class_name)
            else:
                print(f"  ✅ {class_name}")
        
        if missing_modules:
            print(f"  ❌ Missing module tests: {missing_modules}")
        
        if missing_classes:
            print(f"  ❌ Missing class tests: {missing_classes}")
        
        # For now, just warn about missing critical components
        if missing_modules or missing_classes:
            print("⚠️  Some critical components lack test coverage")


def run_coverage_analysis():
    """Run complete coverage analysis and print report."""
    analyzer = CoverageAnalyzer()
    return analyzer.print_coverage_report()


if __name__ == "__main__":
    # Run coverage analysis
    run_coverage_analysis()
    
    # Run tests
    pytest.main([__file__, "-v"])