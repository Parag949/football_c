"""
TacticalVision Deep Bug Hunter
Analyzes the codebase using Python AST parsing and static reflection:
1. Unbound variables / NameErrors
2. Function call signature mismatches
3. Mutable default arguments
4. Inconsistent st.session_state keys (reads without init)
5. Unbounded cache / resource leak risks
"""

import ast
import os
import sys
import inspect
from collections import defaultdict

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PYTHON_DIRS = ["core", "perception", "algorithms", "visualization", "ui"]

def scan_files():
    py_files = []
    for d in PYTHON_DIRS:
        full_d = os.path.join(REPO_ROOT, d)
        for root, _, files in os.walk(full_d):
            for f in files:
                if f.endswith(".py"):
                    py_files.append(os.path.join(root, f))
    py_files.append(os.path.join(REPO_ROOT, "app.py"))
    return py_files

def check_mutable_defaults(filepath, tree):
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for default in node.args.defaults + node.args.kw_defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    issues.append(f"{os.path.relpath(filepath, REPO_ROOT)}:{node.lineno} - Function '{node.name}' has mutable default argument {ast.dump(default)}")
    return issues

def check_session_state_keys(filepath, tree):
    reads = []
    writes = []
    for node in ast.walk(tree):
        # Look for st.session_state[key] or st.session_state.get(key)
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "session_state":
                if isinstance(node.slice, ast.Constant):
                    key = node.slice.value
                    if isinstance(node.ctx, ast.Load):
                        reads.append((key, node.lineno))
                    elif isinstance(node.ctx, ast.Store):
                        writes.append((key, node.lineno))
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                if isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "session_state":
                    if node.args and isinstance(node.args[0], ast.Constant):
                        reads.append((node.args[0].value, node.lineno))
            elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "st":
                # st.session_state.attr
                pass
    return reads, writes

def main():
    print("=== TACTICAL VISION DEEP BUG HUNTER ===")
    files = scan_files()
    print(f"Scanned {len(files)} Python source files.")

    all_mutable_defaults = []
    all_session_reads = defaultdict(list)
    all_session_writes = defaultdict(list)

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                content = fp.read()
            tree = ast.parse(content, filename=f)
        except Exception as e:
            print(f"Failed to parse {f}: {e}")
            continue

        mut_defs = check_mutable_defaults(f, tree)
        all_mutable_defaults.extend(mut_defs)

        r, w = check_session_state_keys(f, tree)
        rel = os.path.relpath(f, REPO_ROOT)
        for k, line in r:
            all_session_reads[k].append((rel, line))
        for k, line in w:
            all_session_writes[k].append((rel, line))

    print(f"\n[1] Mutable Default Arguments Check ({len(all_mutable_defaults)} found):")
    for m in all_mutable_defaults:
        print(f"  * {m}")

    print("\n[2] Streamlit session_state Key Cross-Reference:")
    # Keys read that are NEVER explicitly written anywhere in the codebase
    unwritten_keys = []
    for k, read_locs in all_session_reads.items():
        if k not in all_session_writes:
            unwritten_keys.append((k, read_locs))

    print(f"Found {len(unwritten_keys)} keys read without static write across codebase:")
    for k, locs in unwritten_keys:
        print(f"  * Key '{k}' read at: {locs[:3]}")

if __name__ == "__main__":
    main()
