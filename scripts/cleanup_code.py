#!/usr/bin/env python3
"""
Code cleanup script for DigiPal project.
Removes unused imports, variables, methods, and files.
Identifies unreachable code flows.
"""

import os
import ast
import sys
from pathlib import Path
from typing import Set, List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CodeAnalyzer(ast.NodeVisitor):
    """AST visitor to analyze Python code for unused elements."""
    
    def __init__(self):
        self.imports = set()
        self.defined_names = set()
        self.used_names = set()
        self.functions = set()
        self.classes = set()
        self.variables = set()
        self.called_functions = set()
        self.accessed_attributes = set()
    
    def visit_Import(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports.add(name)
            self.defined_names.add(name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        for alias in node.names:
            name = alias.asname if alias.asname else alias.name
            self.imports.add(name)
            self.defined_names.add(name)
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        self.functions.add(node.name)
        self.defined_names.add(node.name)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node):
        self.functions.add(node.name)
        self.defined_names.add(node.name)
        self.generic_visit(node)
    
    def visit_ClassDef(self, node):
        self.classes.add(node.name)
        self.defined_names.add(node.name)
        self.generic_visit(node)
    
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.variables.add(target.id)
                self.defined_names.add(target.id)
        self.generic_visit(node)
    
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            self.used_names.add(node.id)
        self.generic_visit(node)
    
    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.called_functions.add(node.func.id)
            self.used_names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.accessed_attributes.add(node.func.attr)
        self.generic_visit(node)
    
    def visit_Attribute(self, node):
        self.accessed_attributes.add(node.attr)
        self.generic_visit(node)


def analyze_file(file_path: Path) -> Dict[str, Any]:
    """Analyze a Python file for unused elements."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        analyzer = CodeAnalyzer()
        analyzer.visit(tree)
        
        # Find unused elements
        unused_imports = analyzer.imports - analyzer.used_names
        unused_functions = analyzer.functions - analyzer.called_functions
        unused_variables = analyzer.variables - analyzer.used_names
        
        return {
            'file': file_path,
            'unused_imports': unused_imports,
            'unused_functions': unused_functions,
            'unused_variables': unused_variables,
            'total_imports': len(analyzer.imports),
            'total_functions': len(analyzer.functions),
            'total_variables': len(analyzer.variables)
        }
    
    except Exception as e:
        logger.error(f"Error analyzing {file_path}: {e}")
        return None


def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in directory."""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'node_modules'}]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    return python_files


def find_unused_files(project_root: Path) -> List[Path]:
    """Find potentially unused files."""
    unused_files = []
    
    # Check for common unused file patterns
    patterns = [
        '**/*_backup.py',
        '**/*_old.py',
        '**/*_temp.py',
        '**/*.pyc',
        '**/*.pyo',
        '**/.DS_Store',
        '**/Thumbs.db'
    ]
    
    for pattern in patterns:
        for file_path in project_root.glob(pattern):
            unused_files.append(file_path)
    
    return unused_files


def check_unreachable_code(file_path: Path) -> List[str]:
    """Check for unreachable code patterns."""
    unreachable_patterns = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Check for code after return statements
            if stripped.startswith('return') and i < len(lines):
                next_line = lines[i].strip() if i < len(lines) else ""
                if next_line and not next_line.startswith(('def ', 'class ', '#', '"""', "'''")):
                    unreachable_patterns.append(f"Line {i+1}: Code after return statement")
            
            # Check for code after raise statements
            if stripped.startswith('raise') and i < len(lines):
                next_line = lines[i].strip() if i < len(lines) else ""
                if next_line and not next_line.startswith(('def ', 'class ', '#', '"""', "'''")):
                    unreachable_patterns.append(f"Line {i+1}: Code after raise statement")
    
    except Exception as e:
        logger.error(f"Error checking unreachable code in {file_path}: {e}")
    
    return unreachable_patterns


def generate_cleanup_report(project_root: Path) -> Dict[str, Any]:
    """Generate comprehensive cleanup report."""
    logger.info("Starting code cleanup analysis...")
    
    python_files = find_python_files(project_root)
    logger.info(f"Found {len(python_files)} Python files")
    
    report = {
        'summary': {
            'total_files': len(python_files),
            'analyzed_files': 0,
            'files_with_issues': 0,
            'total_unused_imports': 0,
            'total_unused_functions': 0,
            'total_unused_variables': 0
        },
        'file_analysis': [],
        'unused_files': [],
        'unreachable_code': []
    }
    
    # Analyze each Python file
    for file_path in python_files:
        analysis = analyze_file(file_path)
        if analysis:
            report['file_analysis'].append(analysis)
            report['summary']['analyzed_files'] += 1
            
            if (analysis['unused_imports'] or 
                analysis['unused_functions'] or 
                analysis['unused_variables']):
                report['summary']['files_with_issues'] += 1
            
            report['summary']['total_unused_imports'] += len(analysis['unused_imports'])
            report['summary']['total_unused_functions'] += len(analysis['unused_functions'])
            report['summary']['total_unused_variables'] += len(analysis['unused_variables'])
            
            # Check for unreachable code
            unreachable = check_unreachable_code(file_path)
            if unreachable:
                report['unreachable_code'].append({
                    'file': file_path,
                    'issues': unreachable
                })
    
    # Find unused files
    report['unused_files'] = find_unused_files(project_root)
    
    return report


def print_cleanup_report(report: Dict[str, Any]):
    """Print formatted cleanup report."""
    print("\n" + "="*60)
    print("🧹 DIGIPAL CODE CLEANUP REPORT")
    print("="*60)
    
    # Summary
    summary = report['summary']
    print(f"\n📊 SUMMARY:")
    print(f"  Total files analyzed: {summary['analyzed_files']}/{summary['total_files']}")
    print(f"  Files with issues: {summary['files_with_issues']}")
    print(f"  Total unused imports: {summary['total_unused_imports']}")
    print(f"  Total unused functions: {summary['total_unused_functions']}")
    print(f"  Total unused variables: {summary['total_unused_variables']}")
    
    # Files with issues
    if summary['files_with_issues'] > 0:
        print(f"\n🔍 FILES WITH UNUSED ELEMENTS:")
        for analysis in report['file_analysis']:
            if (analysis['unused_imports'] or 
                analysis['unused_functions'] or 
                analysis['unused_variables']):
                
                rel_path = os.path.relpath(analysis['file'])
                print(f"\n  📄 {rel_path}")
                
                if analysis['unused_imports']:
                    print(f"    🔗 Unused imports: {', '.join(analysis['unused_imports'])}")
                
                if analysis['unused_functions']:
                    print(f"    🔧 Unused functions: {', '.join(analysis['unused_functions'])}")
                
                if analysis['unused_variables']:
                    print(f"    📝 Unused variables: {', '.join(analysis['unused_variables'])}")
    
    # Unreachable code
    if report['unreachable_code']:
        print(f"\n⚠️  UNREACHABLE CODE:")
        for item in report['unreachable_code']:
            rel_path = os.path.relpath(item['file'])
            print(f"\n  📄 {rel_path}")
            for issue in item['issues']:
                print(f"    ❌ {issue}")
    
    # Unused files
    if report['unused_files']:
        print(f"\n🗑️  POTENTIALLY UNUSED FILES:")
        for file_path in report['unused_files']:
            rel_path = os.path.relpath(file_path)
            print(f"    📄 {rel_path}")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if summary['total_unused_imports'] > 0:
        print("  • Remove unused imports to reduce memory usage and improve load times")
    if summary['total_unused_functions'] > 0:
        print("  • Remove or document unused functions to reduce code complexity")
    if summary['total_unused_variables'] > 0:
        print("  • Remove unused variables to improve code readability")
    if report['unreachable_code']:
        print("  • Fix unreachable code to prevent confusion and potential bugs")
    if report['unused_files']:
        print("  • Review and remove unused files to reduce repository size")
    
    if (summary['total_unused_imports'] == 0 and 
        summary['total_unused_functions'] == 0 and 
        summary['total_unused_variables'] == 0 and 
        not report['unreachable_code'] and 
        not report['unused_files']):
        print("  ✅ Code is clean! No issues found.")
    
    print("\n" + "="*60)


def main():
    """Main function to run code cleanup analysis."""
    project_root = Path(__file__).parent.parent
    
    print("🧹 DigiPal Code Cleanup Analysis")
    print(f"📁 Project root: {project_root}")
    
    # Generate report
    report = generate_cleanup_report(project_root)
    
    # Print report
    print_cleanup_report(report)
    
    # Save report to file
    import json
    report_file = project_root / "cleanup_report.json"
    
    # Convert Path objects to strings for JSON serialization
    json_report = report.copy()
    json_report['file_analysis'] = [
        {**analysis, 'file': str(analysis['file'])} 
        for analysis in report['file_analysis']
    ]
    json_report['unused_files'] = [str(f) for f in report['unused_files']]
    json_report['unreachable_code'] = [
        {**item, 'file': str(item['file'])} 
        for item in report['unreachable_code']
    ]
    
    with open(report_file, 'w') as f:
        json.dump(json_report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")


if __name__ == "__main__":
    main()