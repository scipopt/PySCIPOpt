#!/usr/bin/env python3
"""
Automatic stub generator for PySCIPOpt.

This script parses .pxi and .pxd files and generates .pyi stub files automatically.
It extracts class definitions, methods, attributes, and module-level functions.

Usage:
    python scripts/generate_stubs.py

The generated stub will be written to src/pyscipopt/scip.pyi
"""

import re
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict


@dataclass
class ClassInfo:
    """Holds information about a class."""
    name: str
    parent: Optional[str] = None
    attributes: list = field(default_factory=list)  # list of (name, type_hint)
    methods: list = field(default_factory=list)  # list of method names
    static_methods: set = field(default_factory=set)  # set of static method names
    class_vars: list = field(default_factory=list)  # list of (name, type_hint)
    has_hash: bool = False
    has_eq: bool = False
    is_dataclass: bool = False
    dataclass_fields: list = field(default_factory=list)  # list of (name, type, default)


@dataclass
class ModuleInfo:
    """Holds information about the module."""
    classes: OrderedDict = field(default_factory=OrderedDict)
    functions: list = field(default_factory=list)
    module_vars: list = field(default_factory=list)


class StubGenerator:
    """Generates Python stub files from Cython .pxi and .pxd files."""

    # Special methods that need specific type hints
    COMPARISON_METHODS = {
        '__eq__': 'def __eq__(self, other: object) -> bool: ...',
        '__ne__': 'def __ne__(self, other: object) -> bool: ...',
        '__lt__': 'def __lt__(self, other: object) -> bool: ...',
        '__le__': 'def __le__(self, other: object) -> bool: ...',
        '__gt__': 'def __gt__(self, other: object) -> bool: ...',
        '__ge__': 'def __ge__(self, other: object) -> bool: ...',
    }

    SPECIAL_METHODS = {
        '__hash__': 'def __hash__(self) -> int: ...',
        '__len__': 'def __len__(self) -> int: ...',
        '__bool__': 'def __bool__(self) -> bool: ...',
        '__init__': 'def __init__(self, *args, **kwargs) -> None: ...',
        '__repr__': 'def __repr__(self) -> str: ...',
        '__str__': 'def __str__(self) -> str: ...',
        '__delitem__': 'def __delitem__(self, other) -> None: ...',
        '__setitem__': 'def __setitem__(self, index, object) -> None: ...',
    }

    # Methods that should NOT appear in stubs (internal Cython methods)
    EXCLUDED_METHODS = {
        '__cinit__', '__dealloc__', '__reduce__', '__reduce_cython__',
        '__setstate_cython__', '__pyx_vtable__', '__repr__', '__str__',
    }

    # Methods that should be expanded to comparison methods
    RICHCMP_EXPANSION = {
        '__richcmp__': ['__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__'],
    }

    # Methods with specific type hints (no *args, **kwargs)
    TYPED_METHODS = {
        '__getitem__': 'def __getitem__(self, index): ...',
        '__iter__': 'def __iter__(self): ...',
        '__next__': 'def __next__(self): ...',
        '__add__': 'def __add__(self, other): ...',
        '__radd__': 'def __radd__(self, other): ...',
        '__sub__': 'def __sub__(self, other): ...',
        '__rsub__': 'def __rsub__(self, other): ...',
        '__mul__': 'def __mul__(self, other): ...',
        '__rmul__': 'def __rmul__(self, other): ...',
        '__truediv__': 'def __truediv__(self, other): ...',
        '__rtruediv__': 'def __rtruediv__(self, other): ...',
        '__pow__': 'def __pow__(self, other): ...',
        '__rpow__': 'def __rpow__(self, other): ...',
        '__neg__': 'def __neg__(self): ...',
        '__abs__': 'def __abs__(self): ...',
        '__iadd__': 'def __iadd__(self, other): ...',
        '__isub__': 'def __isub__(self, other): ...',
        '__imul__': 'def __imul__(self, other): ...',
    }

    def __init__(self, src_dir: Path):
        self.src_dir = src_dir
        self.module_info = ModuleInfo()

        # Classes that inherit from numpy.ndarray need special handling
        self.numpy_classes = {'MatrixExpr', 'MatrixConstraint'}

        # Known parent classes mapping (from class name to parent in stubs)
        self.inheritance_map = {
            'Variable': 'Expr',
            'Constant': 'GenExpr',
            'VarExpr': 'GenExpr',
            'PowExpr': 'GenExpr',
            'UnaryExpr': 'GenExpr',
            'SumExpr': 'GenExpr',
            'ProdExpr': 'GenExpr',
            'MatrixExpr': 'numpy.ndarray',
            'MatrixConstraint': 'numpy.ndarray',
            'MatrixExprCons': 'numpy.ndarray',
            'MatrixGenExpr': 'MatrixExpr',
            'MatrixVariable': 'MatrixExpr',
        }

    def parse_pxi_file(self, filepath: Path) -> None:
        """Parse a .pxi file and extract class/function definitions."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        self._parse_content(content, filepath)

    def parse_pxd_file(self, filepath: Path) -> None:
        """Parse a .pxd file and extract public attributes."""
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find class definitions and their public attributes
        class_pattern = re.compile(
            r'^cdef\s+class\s+(\w+)(?:\s*\((\w+)\))?:\s*$',
            re.MULTILINE
        )

        attr_pattern = re.compile(
            r'^\s+cdef\s+public\s+(?:object\s+)?(\w+)\s*$',
            re.MULTILINE
        )

        lines = content.split('\n')
        current_class = None

        for i, line in enumerate(lines):
            # Check for class definition
            class_match = class_pattern.match(line)
            if class_match:
                class_name = class_match.group(1)
                parent = class_match.group(2)
                current_class = class_name

                if class_name not in self.module_info.classes:
                    self.module_info.classes[class_name] = ClassInfo(
                        name=class_name,
                        parent=parent
                    )
                continue

            # Check for public attribute
            if current_class and 'cdef public' in line:
                # Extract attribute name
                match = re.search(r'cdef\s+public\s+(?:\w+\s+)?(\w+)\s*$', line.strip())
                if match:
                    attr_name = match.group(1)
                    if current_class in self.module_info.classes:
                        cls_info = self.module_info.classes[current_class]
                        if (attr_name, 'Incomplete') not in cls_info.attributes:
                            cls_info.attributes.append((attr_name, 'Incomplete'))

            # Detect end of class (unindented line that's not empty or comment)
            if current_class and line and not line.startswith(' ') and not line.startswith('\t'):
                if not line.startswith('#') and not line.startswith('cdef class'):
                    current_class = None

    def _parse_content(self, content: str, filepath: Path) -> None:
        """Parse content from a .pxi file."""
        lines = content.split('\n')

        # Stack of (class_name, base_indent) for nested class tracking
        # Only track top-level classes (indent 0) for stub generation
        class_stack = []
        in_property = False
        property_name = None

        def get_current_class():
            """Get the current class if we're directly in a top-level class.

            Returns None if:
            - We're not in any class
            - We're in a nested class (not the top-level class)
            """
            if not class_stack:
                return None
            # Only return a class if we're directly in a top-level (indent 0) class
            # If the latest class on the stack is not at indent 0, we're in a nested class
            latest_class, latest_indent = class_stack[-1]
            if latest_indent == 0:
                return latest_class
            return None

        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines and comments
            if not stripped or stripped.startswith('#'):
                i += 1
                continue

            # Calculate indentation (handle tabs as 4 spaces)
            raw_indent = 0
            for ch in line:
                if ch == ' ':
                    raw_indent += 1
                elif ch == '\t':
                    raw_indent += 4
                else:
                    break
            indent = raw_indent

            # Pop classes from stack when we see a line at or below their base indent
            while class_stack and indent <= class_stack[-1][1]:
                class_stack.pop()

            # Check for class definition
            class_match = re.match(
                r'^(?:cdef\s+)?class\s+(\w+)(?:\s*\(([^)]*)\))?:\s*$',
                stripped
            )
            if class_match:
                class_name = class_match.group(1)
                parent = class_match.group(2)

                # Clean up parent (remove Cython types)
                if parent:
                    parent = parent.strip()
                    # Skip internal Cython parent classes
                    if parent in ('object',):
                        parent = None

                # Push to class stack
                class_stack.append((class_name, indent))

                # Only add top-level classes (indent 0) to module info
                if indent == 0:
                    if class_name not in self.module_info.classes:
                        self.module_info.classes[class_name] = ClassInfo(
                            name=class_name,
                            parent=self.inheritance_map.get(class_name, parent)
                        )
                i += 1
                continue

            current_class = get_current_class()

            # Check for property definition
            property_match = re.match(r'^property\s+(\w+)\s*:', stripped)
            if property_match and current_class:
                property_name = property_match.group(1)
                in_property = True
                i += 1
                continue

            # Check for property __get__ (indicates a readable attribute)
            if in_property and 'def __get__(self)' in stripped:
                if current_class and property_name:
                    cls_info = self.module_info.classes.get(current_class)
                    if cls_info and (property_name, 'Incomplete') not in cls_info.attributes:
                        cls_info.attributes.append((property_name, 'Incomplete'))
                in_property = False
                property_name = None
                i += 1
                continue

            # Check for public attribute in class body
            if current_class and 'cdef public' in stripped:
                match = re.search(r'cdef\s+public\s+(?:\w+\s+)?(\w+)\s*$', stripped)
                if match:
                    attr_name = match.group(1)
                    cls_info = self.module_info.classes.get(current_class)
                    if cls_info and (attr_name, 'Incomplete') not in cls_info.attributes:
                        cls_info.attributes.append((attr_name, 'Incomplete'))
                i += 1
                continue

            # Check for __slots__ definition
            if current_class and '__slots__' in stripped:
                slots_match = re.search(r"__slots__\s*=\s*\(([^)]+)\)", stripped)
                if slots_match:
                    slots_str = slots_match.group(1)
                    # Parse slot names
                    slot_names = re.findall(r"['\"](\w+)['\"]", slots_str)
                    cls_info = self.module_info.classes.get(current_class)
                    if cls_info:
                        for slot_name in slot_names:
                            if (slot_name, 'Incomplete') not in cls_info.attributes:
                                cls_info.attributes.append((slot_name, 'Incomplete'))
                i += 1
                continue

            # Check for class variable assignment (for enum-like classes)
            if current_class:
                cls_info = self.module_info.classes.get(current_class)
                if cls_info:
                    # Handle multi-assignment lines like: exp, log, sqrt = 'exp', 'log', 'sqrt'
                    # Only for Op class (special case for operator definitions)
                    if current_class == 'Op' and indent == 4:
                        multi_match = re.match(r'^(\w+(?:\s*,\s*\w+)+)\s*=\s*(.+)$', stripped)
                        if multi_match:
                            var_names = [v.strip() for v in multi_match.group(1).split(',')]
                            var_values = multi_match.group(2).strip()

                            # Determine type from first value
                            if var_values.startswith('"') or var_values.startswith("'"):
                                type_hint = 'str'
                            elif var_values.isdigit() or 'SCIP_' in var_values:
                                type_hint = 'int'
                            else:
                                type_hint = 'str'

                            for var_name in var_names:
                                if not var_name.startswith('_') and var_name not in ('self',):
                                    if (var_name, type_hint) not in cls_info.class_vars:
                                        cls_info.class_vars.append((var_name, type_hint))
                            i += 1
                            continue

                    # Match pattern like: ATTRNAME = SCIP_VALUE or ATTRNAME = VALUE
                    var_match = re.match(r'^(\w+)\s*=\s*(.+)$', stripped)
                    if var_match:
                        var_name = var_match.group(1)
                        var_value = var_match.group(2).strip()

                        # Skip private variables, special cases, and method assignments
                        if not var_name.startswith('_') and var_name not in ('self',):
                            # Determine type from value pattern
                            if 'SCIP_' in var_value or var_value.isdigit():
                                type_hint = 'int'
                            elif var_value.startswith('"') or var_value.startswith("'"):
                                type_hint = 'str'
                            else:
                                type_hint = 'int'  # default for enum-like

                            # For UPPERCASE names, use ClassVar
                            if var_name.isupper():
                                if (var_name, type_hint) not in cls_info.class_vars:
                                    cls_info.class_vars.append((var_name, type_hint))
                            # For lowercase class variables (like in Op class), only add if:
                            # - Class has no methods yet (we're at class-level, not in a method)
                            # - The class is in a known list of classes with class vars
                            elif current_class == 'Op' and indent == 4:
                                if (var_name, type_hint) not in cls_info.class_vars:
                                    cls_info.class_vars.append((var_name, type_hint))
                        i += 1
                        continue

            # Check for decorators (look at previous non-empty lines)
            is_property = False
            is_setter = False
            is_staticmethod = False
            if stripped.startswith('def ') and i > 0:
                # Look back for decorators
                for j in range(i - 1, max(0, i - 5), -1):
                    prev_line = lines[j].strip()
                    if prev_line == '@property':
                        is_property = True
                    elif '.setter' in prev_line or '.deleter' in prev_line:
                        is_setter = True
                    elif prev_line == '@staticmethod':
                        is_staticmethod = True
                    elif prev_line and not prev_line.startswith('#') and not prev_line.startswith('@'):
                        break

            # Check for method definition (handle multi-line signatures)
            # First, check if this line starts a def/cpdef
            method_start_match = re.match(
                r'^(?:cpdef|def)\s+(\w+)\s*\(',
                stripped
            )
            if method_start_match:
                method_name = method_start_match.group(1)

                # Skip setters (they don't need separate stubs)
                if is_setter:
                    i += 1
                    continue

                # Skip nested functions (defined inside methods, at high indentation)
                # Methods should be at indent 4 for top-level class methods
                if current_class and indent > 4:
                    # This is likely a nested function, skip it
                    i += 1
                    continue

                # Skip excluded methods (internal Cython methods)
                if method_name in self.EXCLUDED_METHODS:
                    i += 1
                    continue

                if current_class:
                    cls_info = self.module_info.classes.get(current_class)
                    if cls_info:
                        if is_property:
                            # Add as attribute instead of method
                            if (method_name, 'Incomplete') not in cls_info.attributes:
                                cls_info.attributes.append((method_name, 'Incomplete'))
                        elif method_name == '__richcmp__':
                            # Expand __richcmp__ to individual comparison methods
                            for cmp_method in self.RICHCMP_EXPANSION['__richcmp__']:
                                if cmp_method not in cls_info.methods:
                                    cls_info.methods.append(cmp_method)
                        elif method_name not in cls_info.methods:
                            cls_info.methods.append(method_name)

                            # Track static methods
                            if is_staticmethod:
                                cls_info.static_methods.add(method_name)

                            # Track special methods
                            if method_name == '__hash__':
                                cls_info.has_hash = True
                            elif method_name == '__eq__':
                                cls_info.has_eq = True
                else:
                    # Module-level function (must be at indent 0)
                    if indent == 0 and method_name not in self.module_info.functions:
                        self.module_info.functions.append(method_name)

                i += 1
                continue

            # Check for module-level variable
            if not current_class and indent == 0:
                var_match = re.match(r'^(\w+)\s*=\s*', stripped)
                if var_match:
                    var_name = var_match.group(1)
                    if not var_name.startswith('_') and var_name not in ('include',):
                        if var_name not in [v[0] for v in self.module_info.module_vars]:
                            self.module_info.module_vars.append((var_name, 'Incomplete'))

            i += 1

    def _detect_dataclass(self, content: str) -> dict:
        """Detect @dataclass decorated classes and their fields."""
        dataclasses = {}

        # Find @dataclass decorated classes
        pattern = re.compile(
            r'@dataclass\s*\n\s*class\s+(\w+)(?:\s*\([^)]*\))?\s*:\s*\n((?:\s+.+\n)*)',
            re.MULTILINE
        )

        for match in pattern.finditer(content):
            class_name = match.group(1)
            body = match.group(2)

            fields = []
            for line in body.split('\n'):
                # Match field: type = default or field: type
                field_match = re.match(r'\s+(\w+)\s*:\s*(\w+)(?:\s*=\s*(.+))?', line)
                if field_match:
                    fname = field_match.group(1)
                    ftype = field_match.group(2)
                    fdefault = field_match.group(3)
                    fields.append((fname, ftype, fdefault))

            dataclasses[class_name] = fields

        return dataclasses

    def generate_stub(self) -> str:
        """Generate the complete stub file content."""
        lines = []

        # Header imports
        lines.append('from dataclasses import dataclass')
        lines.append('from typing import ClassVar')
        lines.append('')
        lines.append('import numpy')
        lines.append('from _typeshed import Incomplete')
        lines.append('')

        # Module-level variables and functions (sorted alphabetically)
        # Merge variables and functions into a single sorted list
        all_module_items = []

        for var_name, type_hint in self.module_info.module_vars:
            all_module_items.append((var_name, 'var', type_hint))

        for func_name in self.module_info.functions:
            all_module_items.append((func_name, 'func', None))

        # Sort by name
        all_module_items.sort(key=lambda x: x[0])

        for name, kind, type_hint in all_module_items:
            if kind == 'var':
                lines.append(f'{name}: {type_hint}')
            else:
                lines.append(f'{name}: Incomplete')

        if all_module_items:
            lines.append('')

        # Classes (sorted alphabetically)
        sorted_classes = sorted(
            self.module_info.classes.values(),
            key=lambda c: c.name
        )

        for cls_info in sorted_classes:
            # Skip internal classes
            if cls_info.name.startswith('_') and cls_info.name != '_VarArray':
                continue

            lines.extend(self._generate_class_stub(cls_info))
            lines.append('')

        return '\n'.join(lines)

    def _generate_class_stub(self, cls_info: ClassInfo) -> list:
        """Generate stub for a single class."""
        lines = []

        # Handle dataclass
        if cls_info.is_dataclass:
            lines.append('@dataclass')

        # Class declaration
        if cls_info.parent:
            lines.append(f'class {cls_info.name}({cls_info.parent}):')
        else:
            lines.append(f'class {cls_info.name}:')

        # Dataclass fields
        if cls_info.dataclass_fields:
            for fname, ftype, fdefault in cls_info.dataclass_fields:
                if fdefault:
                    lines.append(f'    {fname}: {ftype} = {fdefault}')
                else:
                    lines.append(f'    {fname}: {ftype}')
            return lines

        # Class variables (for enum-like classes)
        sorted_class_vars = sorted(cls_info.class_vars, key=lambda x: x[0])
        for var_name, type_hint in sorted_class_vars:
            lines.append(f'    {var_name}: ClassVar[{type_hint}] = ...')

        # Instance attributes
        sorted_attrs = sorted(cls_info.attributes, key=lambda x: x[0])
        for attr_name, type_hint in sorted_attrs:
            lines.append(f'    {attr_name}: {type_hint}')

        # Methods - separate regular methods and special methods
        regular_methods = []
        special_methods = []
        comparison_methods = []

        for method in cls_info.methods:
            if method in self.COMPARISON_METHODS:
                comparison_methods.append(method)
            elif method.startswith('__') and method.endswith('__'):
                special_methods.append(method)
            else:
                regular_methods.append(method)

        # Check if this class inherits from numpy.ndarray or MatrixExpr (don't auto-add __init__)
        is_numpy_subclass = cls_info.parent and ('ndarray' in cls_info.parent or 'Matrix' in cls_info.parent)

        # Specific classes that shouldn't get __init__ auto-added
        skip_init_classes = {'Op'}

        # If class has both __hash__ and __eq__, add all comparison methods
        if cls_info.has_hash and cls_info.has_eq:
            for cmp_method in self.COMPARISON_METHODS:
                if cmp_method not in comparison_methods and cmp_method not in special_methods:
                    comparison_methods.append(cmp_method)

        # Output __init__ first if present or needs to be added (not for numpy subclasses or specific classes)
        if '__init__' in special_methods:
            lines.append(f'    {self.SPECIAL_METHODS["__init__"]}')
            special_methods.remove('__init__')
        elif '__init__' not in cls_info.methods and not is_numpy_subclass and not cls_info.is_dataclass and cls_info.name not in skip_init_classes:
            lines.append(f'    {self.SPECIAL_METHODS["__init__"]}')

        # Sort and output regular methods
        for method in sorted(regular_methods):
            if method in cls_info.static_methods:
                lines.append('    @staticmethod')
                lines.append(f'    def {method}(*args, **kwargs): ...')
            else:
                lines.append(f'    def {method}(self, *args, **kwargs): ...')

        # Combine special methods and comparison methods, sort alphabetically
        all_special = []
        for method in special_methods:
            if method in self.SPECIAL_METHODS:
                all_special.append((method, self.SPECIAL_METHODS[method]))
            elif method in self.TYPED_METHODS:
                all_special.append((method, self.TYPED_METHODS[method]))
            else:
                all_special.append((method, f'def {method}(self, *args, **kwargs): ...'))

        for method in comparison_methods:
            all_special.append((method, self.COMPARISON_METHODS[method]))

        # Sort and output all special methods alphabetically
        for method, stub in sorted(all_special, key=lambda x: x[0]):
            lines.append(f'    {stub}')

        # If class is empty, add pass or ellipsis
        if len(lines) == 1:
            lines.append('    ...')

        return lines

    def run(self) -> str:
        """Run the stub generator on all relevant files."""
        pxi_files = list(self.src_dir.glob('*.pxi'))
        pxd_files = list(self.src_dir.glob('*.pxd'))

        # Parse .pxd files first for attribute declarations
        for pxd_file in pxd_files:
            print(f'Parsing {pxd_file.name}...')
            self.parse_pxd_file(pxd_file)

        # Parse .pxi files for implementations
        for pxi_file in pxi_files:
            print(f'Parsing {pxi_file.name}...')
            self.parse_pxi_file(pxi_file)

        # Handle special cases
        self._apply_special_cases()

        # Generate the stub
        return self.generate_stub()

    def _apply_special_cases(self) -> None:
        """Apply special cases and known patterns."""
        # Add 'name' attribute to classes that commonly have it
        name_classes = {
            'Variable', 'Constraint', 'Row', 'NLRow', 'Event',
            'Benders', 'Benderscut', 'Conshdlr', 'Heur', 'Sepa',
            'Reader', 'Relax', 'Eventhdlr'
        }

        for class_name in name_classes:
            if class_name in self.module_info.classes:
                cls_info = self.module_info.classes[class_name]
                if ('name', 'Incomplete') not in cls_info.attributes:
                    cls_info.attributes.append(('name', 'Incomplete'))

        # Handle LP class special case (readonly name)
        if 'LP' in self.module_info.classes:
            cls_info = self.module_info.classes['LP']
            if ('name', 'Incomplete') not in cls_info.attributes:
                cls_info.attributes.append(('name', 'Incomplete'))

        # Handle Statistics as dataclass
        if 'Statistics' in self.module_info.classes:
            cls_info = self.module_info.classes['Statistics']
            cls_info.is_dataclass = True
            cls_info.dataclass_fields = [
                ('status', 'str', None),
                ('total_time', 'float', None),
                ('solving_time', 'float', None),
                ('presolving_time', 'float', None),
                ('reading_time', 'float', None),
                ('copying_time', 'float', None),
                ('problem_name', 'str', None),
                ('presolved_problem_name', 'str', None),
                ('n_runs', 'int', 'None'),
                ('n_nodes', 'int', 'None'),
                ('n_solutions_found', 'int', '-1'),
                ('first_solution', 'float', 'None'),
                ('primal_bound', 'float', 'None'),
                ('dual_bound', 'float', 'None'),
                ('gap', 'float', 'None'),
            ]

        # Add/update known module-level variables with correct types
        known_vars = {
            'CONST': 'Term',
            'EventNames': 'dict',
            'MAJOR': 'int',
            'MINOR': 'int',
            'Operator': 'Op',
            'PATCH': 'int',
            'StageNames': 'dict',
            '_SCIP_BOUNDTYPE_TO_STRING': 'dict',
            'str_conversion': 'Incomplete',
        }

        # Remove PY_SCIP_CALL from functions if it exists (it's also a callable)
        if 'PY_SCIP_CALL' in self.module_info.functions:
            self.module_info.functions.remove('PY_SCIP_CALL')
            known_vars['PY_SCIP_CALL'] = 'Incomplete'

        # Update existing or add new
        updated_vars = []
        seen = set()
        for var_name, type_hint in self.module_info.module_vars:
            if var_name in known_vars:
                updated_vars.append((var_name, known_vars[var_name]))
                seen.add(var_name)
            else:
                updated_vars.append((var_name, type_hint))
                seen.add(var_name)

        # Add any known vars that weren't in the list
        for var_name, type_hint in known_vars.items():
            if var_name not in seen:
                updated_vars.append((var_name, type_hint))

        self.module_info.module_vars = updated_vars


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Generate Python stub files (.pyi) from PySCIPOpt Cython sources.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python scripts/generate_stubs.py              # Generate and write stubs
  python scripts/generate_stubs.py --check      # Check if stubs are up-to-date
  python scripts/generate_stubs.py --dry-run    # Show what would be generated
'''
    )
    parser.add_argument(
        '--check', action='store_true',
        help='Check if the stub file is up-to-date (exits with 1 if not)'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print the generated stubs to stdout instead of writing to file'
    )
    parser.add_argument(
        '-q', '--quiet', action='store_true',
        help='Suppress progress output'
    )
    args = parser.parse_args()

    # Find the source directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    src_dir = project_root / 'src' / 'pyscipopt'

    if not src_dir.exists():
        print(f'Error: Source directory not found: {src_dir}')
        return 1

    if not args.quiet:
        print(f'Generating stubs from {src_dir}...')

    generator = StubGenerator(src_dir)
    stub_content = generator.run()

    output_path = src_dir / 'scip.pyi'

    if args.dry_run:
        print(stub_content)
        return 0

    if args.check:
        # Compare with existing file
        if output_path.exists():
            with open(output_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
            if existing_content == stub_content:
                if not args.quiet:
                    print('Stub file is up-to-date.')
                return 0
            else:
                print(f'Stub file {output_path} is out of date.')
                print('Run without --check to regenerate.')
                return 1
        else:
            print(f'Stub file {output_path} does not exist.')
            return 1

    # Write the stub file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(stub_content)

    if not args.quiet:
        print(f'Stub file written to {output_path}')
        print(f'\nSummary:')
        print(f'  Classes: {len(generator.module_info.classes)}')
        print(f'  Functions: {len(generator.module_info.functions)}')
        print(f'  Module variables: {len(generator.module_info.module_vars)}')

    return 0


if __name__ == '__main__':
    exit(main())
