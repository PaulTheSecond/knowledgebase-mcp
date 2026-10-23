"""
Модуль для извлечения символов и связей из исходного кода.
Использует Tree-sitter и ast для полиглотного AST парсинга (C#, TypeScript, JavaScript, Python, SQL).
"""

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tree_sitter
import tree_sitter_c_sharp
import tree_sitter_typescript
import tree_sitter_javascript
import tree_sitter_sql

logger = logging.getLogger(__name__)

# Маппинг языков Tree-sitter
TS_LANGUAGES = {
    'c_sharp': tree_sitter.Language(tree_sitter_c_sharp.language()),
    'typescript': tree_sitter.Language(tree_sitter_typescript.language_typescript()),
    'javascript': tree_sitter.Language(tree_sitter_javascript.language()),
    'sql': tree_sitter.Language(tree_sitter_sql.language()),
}

# Маппинг расширений файлов на языки
LANGUAGE_MAP = {
    '.cs': 'c_sharp',
    '.ts': 'typescript', 
    '.tsx': 'typescript',
    '.js': 'javascript', 
    '.jsx': 'javascript',
    '.py': 'python',
    '.sql': 'sql',
}


def extract_tested_target(test_name: str) -> Optional[str]:
    """Извлекает имя тестируемой сущности из имени теста по эвристикам.
    Только если test_name действительно является тестовым именем!
    """
    if not test_name:
        return None
    if test_name.startswith('test_'):
        return test_name[5:]
    if test_name.startswith('Test_'):
        return test_name[5:]
    if test_name.startswith('Test') and len(test_name) > 4 and test_name[4].isupper():
        return test_name[4:]
    if test_name.endswith('Tests'):
        return test_name[:-5]
    if test_name.endswith('TestCase'):
        return test_name[:-8]
    if test_name.endswith('Test') and len(test_name) > 4:
        return test_name[:-4]
    if '_Should' in test_name or '_should' in test_name:
        return test_name.split('_')[0]
    return None


def is_test_name(name: str) -> bool:
    """Проверяет, соответствует ли имя функции или класса шаблону имени тестов."""
    if not name:
        return False
    return name.startswith(('test_', 'Test_')) or name.endswith(('_test', '_tests', 'Test', 'Tests'))

@dataclass
class Symbol:
    name: str              # Имя символа: "UserService"
    qualified_name: str    # Полное имя: "Yurta.Core.Services.UserService"
    kind: str              # Тип: class, method, function, interface, property, enum, test...
    language: str          # c_sharp, typescript, python...
    line_start: int        # 1-indexed
    line_end: int          # 1-indexed
    signature: str         # Сигнатура: "public async Task<User> GetById(int id)"
    body: str              # Полный текст узла для чанка
    is_test: bool = False  # Флаг: является ли символ тестовым методом/классом


@dataclass
class SymbolEdge:
    source_name: str       # Имя/qualified_name вызывающего (кто)
    target_name: str       # Имя/qualified_name вызываемого (кого)
    kind: str              # CALLS, INHERITS, IMPLEMENTS, IMPORTS, USES, TESTED_BY


class CodeParser:
    def __init__(self):
        self.parsers = {}
        for lang_name, lang_obj in TS_LANGUAGES.items():
            parser = tree_sitter.Parser(lang_obj)
            self.parsers[lang_name] = parser

    def parse_file(self, filepath: str | Path, language: str) -> list[Symbol]:
        """Извлекает все символы из файла."""
        filepath = Path(filepath)
        if language == 'python':
            try:
                content = filepath.read_bytes()
                return self._parse_python_symbols(content)
            except Exception as e:
                logger.error(f"Failed to parse python symbols in {filepath}: {e}")
                return []

        if language not in self.parsers:
            logger.warning(f"Language {language} not supported by CodeParser.")
            return []

        try:
            content = filepath.read_bytes()
            tree = self.parsers[language].parse(content)
            
            if language == 'c_sharp':
                return self._parse_csharp_symbols(tree, content)
            elif language in ('typescript', 'javascript'):
                return self._parse_ts_js_symbols(tree, content, language)
            elif language == 'sql':
                return self._parse_sql_symbols(tree, content)
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to parse symbols in {filepath}: {e}")
            return []

    def extract_edges(self, filepath: str | Path, language: str) -> list[SymbolEdge]:
        """Извлекает связи (calls, imports, inherits, TESTED_BY) из файла."""
        filepath = Path(filepath)
        if language == 'python':
            try:
                content = filepath.read_bytes()
                return self._extract_python_edges(content)
            except Exception as e:
                logger.error(f"Failed to extract python edges in {filepath}: {e}")
                return []

        if language not in self.parsers:
            return []

        try:
            content = filepath.read_bytes()
            tree = self.parsers[language].parse(content)
            
            if language == 'c_sharp':
                return self._extract_csharp_edges(tree, content)
            elif language in ('typescript', 'javascript'):
                return self._extract_ts_js_edges(tree, content, language)
            elif language == 'sql':
                return self._extract_sql_edges(tree, content)
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to extract edges in {filepath}: {e}")
            return []

    def _get_node_text(self, node: tree_sitter.Node, content: bytes) -> str:
        return content[node.start_byte:node.end_byte].decode('utf-8', errors='replace')

    # -------------------------------------------------------------------------
    # Python Parsing (через встроенный модуль ast)
    # -------------------------------------------------------------------------
    def _parse_python_symbols(self, content: bytes) -> list[Symbol]:
        """Парсинг Python файлов через ast (чистая функция)."""
        symbols = []
        try:
            source = content.decode('utf-8', errors='replace')
            tree = ast.parse(source)
            lines = source.splitlines()
        except Exception as e:
            logger.warning(f"Failed to parse python AST: {e}")
            return []

        def get_body(start_line: int, end_line: int) -> str:
            return "\n".join(lines[start_line - 1 : end_line])

        def is_test_name(name: str) -> bool:
            return name.startswith(('test_', 'Test_')) or name.endswith(('_test', '_tests', 'Test', 'Tests'))

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                cls_is_test = is_test_name(node.name)
                for base in node.bases:
                    if isinstance(base, ast.Name) and 'Test' in base.id:
                        cls_is_test = True
                    elif isinstance(base, ast.Attribute) and 'TestCase' in base.attr:
                        cls_is_test = True

                kind = 'test' if cls_is_test else 'class'
                line_start = node.lineno
                line_end = getattr(node, 'end_lineno', node.lineno)
                body = get_body(line_start, line_end)
                signature = f"class {node.name}"
                symbols.append(Symbol(
                    name=node.name,
                    qualified_name=node.name,
                    kind=kind,
                    language='python',
                    line_start=line_start,
                    line_end=line_end,
                    signature=signature,
                    body=body,
                    is_test=cls_is_test
                ))

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_is_test = cls_is_test or is_test_name(item.name)
                        m_kind = 'test' if method_is_test else 'method'
                        m_start = item.lineno
                        m_end = getattr(item, 'end_lineno', item.lineno)
                        m_body = get_body(m_start, m_end)
                        qname = f"{node.name}.{item.name}"
                        args_list = [a.arg for a in item.args.args]
                        m_sig = f"def {item.name}({', '.join(args_list)})"
                        symbols.append(Symbol(
                            name=item.name,
                            qualified_name=qname,
                            kind=m_kind,
                            language='python',
                            line_start=m_start,
                            line_end=m_end,
                            signature=m_sig,
                            body=m_body,
                            is_test=method_is_test
                        ))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fn_is_test = is_test_name(node.name)
                kind = 'test' if fn_is_test else 'function'
                line_start = node.lineno
                line_end = getattr(node, 'end_lineno', node.lineno)
                body = get_body(line_start, line_end)
                args_list = [a.arg for a in node.args.args]
                signature = f"def {node.name}({', '.join(args_list)})"
                symbols.append(Symbol(
                    name=node.name,
                    qualified_name=node.name,
                    kind=kind,
                    language='python',
                    line_start=line_start,
                    line_end=line_end,
                    signature=signature,
                    body=body,
                    is_test=fn_is_test
                ))

        return symbols

    def _extract_python_edges(self, content: bytes) -> list[SymbolEdge]:
        """Извлечение связей (CALLS, INHERITS, IMPORTS, TESTED_BY) из Python AST."""
        edges = []
        try:
            source = content.decode('utf-8', errors='replace')
            tree = ast.parse(source)
        except Exception:
            return []

        def get_func_name(func_node: ast.AST) -> Optional[str]:
            if isinstance(func_node, ast.Name):
                return func_node.id
            elif isinstance(func_node, ast.Attribute):
                return func_node.attr
            return None

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        edges.append(SymbolEdge("file_level", alias.name, "IMPORTS"))
                else:
                    mod = node.module or ""
                    for alias in node.names:
                        edges.append(SymbolEdge("file_level", f"{mod}.{alias.name}" if mod else alias.name, "IMPORTS"))

            elif isinstance(node, ast.ClassDef):
                for base in node.bases:
                    base_name = get_func_name(base)
                    if base_name:
                        edges.append(SymbolEdge(node.name, base_name, "INHERITS"))

                cls_is_test = is_test_name(node.name)
                for base in node.bases:
                    if isinstance(base, ast.Name) and 'Test' in base.id:
                        cls_is_test = True
                    elif isinstance(base, ast.Attribute) and 'TestCase' in base.attr:
                        cls_is_test = True

                if cls_is_test:
                    target_candidate = extract_tested_target(node.name)
                    if target_candidate and target_candidate != node.name:
                        edges.append(SymbolEdge(target_candidate, node.name, "TESTED_BY"))

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        qname = f"{node.name}.{item.name}"
                        method_is_test = cls_is_test or is_test_name(item.name)
                        for subnode in ast.walk(item):
                            if isinstance(subnode, ast.Call):
                                called = get_func_name(subnode.func)
                                if called and called != item.name:
                                    edges.append(SymbolEdge(qname, called, "CALLS"))

                        if method_is_test:
                            m_target = extract_tested_target(item.name)
                            if m_target and m_target != item.name:
                                edges.append(SymbolEdge(m_target, qname, "TESTED_BY"))

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Call):
                        called = get_func_name(subnode.func)
                        if called and called != node.name:
                            edges.append(SymbolEdge(node.name, called, "CALLS"))

                fn_is_test = is_test_name(node.name)
                if fn_is_test:
                    fn_target = extract_tested_target(node.name)
                    if fn_target and fn_target != node.name:
                        edges.append(SymbolEdge(fn_target, node.name, "TESTED_BY"))

        return edges

    # -------------------------------------------------------------------------
    # C# Parsing
    # -------------------------------------------------------------------------
    def _parse_csharp_symbols(self, tree: tree_sitter.Tree, content: bytes) -> list[Symbol]:
        symbols = []
        
        def check_csharp_is_test(node: tree_sitter.Node, name: str) -> bool:
            if name.endswith(('Tests', 'Test', 'TestCase')) or name.startswith(('Test_', 'Test', 'test_')):
                return True
            for child in node.children:
                if child.type == 'attribute_list':
                    for attr in child.children:
                        if attr.type == 'attribute':
                            for a_child in attr.children:
                                if a_child.type == 'identifier' and self._get_node_text(a_child, content) in ('Fact', 'Test', 'TestCase', 'Theory', 'TestMethod'):
                                    return True
            return False

        def traverse(node: tree_sitter.Node, parent_qualified_name: str):
            current_qualified_name = parent_qualified_name
            
            if node.type in ('namespace_declaration', 'file_scoped_namespace_declaration'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    ns_name = self._get_node_text(name_node, content)
                    current_qualified_name = ns_name if not parent_qualified_name else f"{parent_qualified_name}.{ns_name}"
            
            elif node.type in ('class_declaration', 'struct_declaration', 'interface_declaration', 'record_declaration', 'enum_declaration'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    kind = node.type.split('_')[0]
                    name = self._get_node_text(name_node, content)
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
                    
                    is_test = check_csharp_is_test(node, name)
                    if is_test:
                        kind = 'test'

                    body_node = node.child_by_field_name('body')
                    if body_node:
                        sig_bytes = content[node.start_byte:body_node.start_byte]
                    else:
                        sig_bytes = content[node.start_byte:node.end_byte]
                    signature = sig_bytes.decode('utf-8', errors='replace').strip()
                    body = self._get_node_text(node, content)
                    
                    symbols.append(Symbol(
                        name=name,
                        qualified_name=current_qualified_name,
                        kind=kind,
                        language='c_sharp',
                        line_start=node.start_point[0] + 1,
                        line_end=node.end_point[0] + 1,
                        signature=signature,
                        body=body,
                        is_test=is_test
                    ))

            elif node.type in ('method_declaration', 'constructor_declaration'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_node_text(name_node, content)
                    kind = 'method' if node.type == 'method_declaration' else 'constructor'
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
                    
                    is_test = check_csharp_is_test(node, name)
                    if is_test:
                        kind = 'test'

                    body_node = node.child_by_field_name('body')
                    if body_node:
                        sig = content[node.start_byte:body_node.start_byte].decode('utf-8', errors='replace').strip()
                    else:
                        sig = content[node.start_byte:node.end_byte].decode('utf-8', errors='replace').strip()

                    symbols.append(Symbol(
                        name=name, qualified_name=current_qualified_name,
                        kind=kind, language='c_sharp',
                        line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                        signature=sig, body=self._get_node_text(node, content),
                        is_test=is_test
                    ))

            elif node.type == 'property_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_node_text(name_node, content)
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
                    
                    accessors = node.child_by_field_name('accessors')
                    if accessors:
                        sig = content[node.start_byte:accessors.start_byte].decode('utf-8', errors='replace').strip()
                    else:
                        sig = content[node.start_byte:node.end_byte].decode('utf-8', errors='replace').strip()

                    symbols.append(Symbol(
                        name=name, qualified_name=current_qualified_name,
                        kind='property', language='c_sharp',
                        line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                        signature=sig, body=self._get_node_text(node, content)
                    ))

            elif node.type == 'field_declaration':
                first_name = None
                for child in node.children:
                    if child.type == 'variable_declaration':
                        for vchild in child.children:
                            if vchild.type == 'variable_declarator':
                                name_node = vchild.child_by_field_name('name')
                                if name_node:
                                    first_name = self._get_node_text(name_node, content)
                                    break
                        if first_name:
                            break
                
                if first_name:
                    current_qualified_name = f"{parent_qualified_name}.{first_name}" if parent_qualified_name else first_name
                    sig = content[node.start_byte:node.end_byte].decode('utf-8', errors='replace').strip()
                    symbols.append(Symbol(
                        name=first_name, qualified_name=current_qualified_name,
                        kind='field', language='c_sharp',
                        line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                        signature=sig, body=self._get_node_text(node, content)
                    ))
            
            elif node.type == 'enum_member_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_node_text(name_node, content)
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
                    symbols.append(Symbol(
                        name=name, qualified_name=current_qualified_name,
                        kind='enum_member', language='c_sharp',
                        line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                        signature=name, body=self._get_node_text(node, content)
                    ))
            
            elif node.type in ('event_declaration', 'event_field_declaration'):
                name_node = node.child_by_field_name('name')
                if not name_node:
                    for child in node.children:
                        if child.type == 'variable_declaration':
                            for vchild in child.children:
                                if vchild.type == 'variable_declarator':
                                    name_node = vchild.child_by_field_name('name')
                                    break
                
                if name_node:
                    name = self._get_node_text(name_node, content)
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
                    sig_node = node.child_by_field_name('accessor_list')
                    end_byte = sig_node.start_byte if sig_node else node.end_byte
                    sig = content[node.start_byte:end_byte].decode('utf-8', errors='replace').strip()
                    symbols.append(Symbol(
                        name=name, qualified_name=current_qualified_name,
                        kind='event', language='c_sharp',
                        line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                        signature=sig, body=self._get_node_text(node, content)
                    ))

            elif node.type == 'delegate_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_node_text(name_node, content)
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
                    sig = self._get_node_text(node, content).strip()
                    symbols.append(Symbol(
                        name=name, qualified_name=current_qualified_name,
                        kind='delegate', language='c_sharp',
                        line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                        signature=sig, body=sig
                    ))

            for child in node.children:
                traverse(child, current_qualified_name)

        traverse(tree.root_node, "")
        return symbols

    def _extract_csharp_edges(self, tree: tree_sitter.Tree, content: bytes) -> list[SymbolEdge]:
        edges = []
        
        def get_identifier_name(node: tree_sitter.Node) -> str:
            if node.type == 'identifier':
                return self._get_node_text(node, content)
            elif node.type == 'member_access_expression':
                name_node = node.child_by_field_name('name')
                if name_node:
                    return self._get_node_text(name_node, content)
            elif node.type == 'generic_name':
                for child in node.children:
                    if child.type == 'identifier':
                        return self._get_node_text(child, content)
            return self._get_node_text(node, content)

        def check_csharp_is_test(node: tree_sitter.Node, name: str) -> bool:
            if name.endswith(('Tests', 'Test', 'TestCase')) or name.startswith(('Test_', 'Test', 'test_')):
                return True
            for child in node.children:
                if child.type == 'attribute_list':
                    for attr in child.children:
                        if attr.type == 'attribute':
                            for a_child in attr.children:
                                if a_child.type == 'identifier' and self._get_node_text(a_child, content) in ('Fact', 'Test', 'TestCase', 'Theory', 'TestMethod'):
                                    return True
            return False

        def traverse(node: tree_sitter.Node, parent_qualified_name: str):
            current_qualified_name = parent_qualified_name
            
            if node.type in ('namespace_declaration', 'file_scoped_namespace_declaration'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    ns_name = self._get_node_text(name_node, content)
                    current_qualified_name = ns_name if not parent_qualified_name else f"{parent_qualified_name}.{ns_name}"
            
            elif node.type in ('class_declaration', 'struct_declaration', 'interface_declaration', 'record_declaration'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_node_text(name_node, content)
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
                
                # Check for base classes/interfaces
                for child in node.children:
                    if child.type == 'base_list':
                        for type_node in child.children:
                            if type_node.type in ('identifier', 'generic_name', 'qualified_name'):
                                target_name = get_identifier_name(type_node)
                                kind = 'IMPLEMENTS' if target_name.startswith('I') and len(target_name) > 1 and target_name[1].isupper() else 'INHERITS'
                                edges.append(SymbolEdge(
                                    source_name=current_qualified_name,
                                    target_name=target_name,
                                    kind=kind
                                ))
            
            elif node.type in ('method_declaration', 'constructor_declaration', 'property_declaration'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_node_text(name_node, content)
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
            
            # Детекция связей TESTED_BY для тестовых классов и методов
            if node.type in ('class_declaration', 'method_declaration'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_node_text(name_node, content)
                    if check_csharp_is_test(node, name):
                        target_candidate = extract_tested_target(name)
                        if target_candidate and target_candidate != name:
                            # source = протестированный символ, target = тест
                            edges.append(SymbolEdge(
                                source_name=target_candidate,
                                target_name=current_qualified_name,
                                kind='TESTED_BY'
                            ))

            # Извлечение связей вызовов и using
            if node.type == 'using_directive':
                for child in node.children:
                    if child.type in ('identifier', 'qualified_name'):
                        target_name = self._get_node_text(child, content)
                        edges.append(SymbolEdge(
                            source_name=current_qualified_name or "file_level", 
                            target_name=target_name,
                            kind='IMPORTS'
                        ))
                        
            elif node.type == 'invocation_expression':
                function_node = node.child_by_field_name('function')
                if function_node:
                    target_name = get_identifier_name(function_node)
                    edges.append(SymbolEdge(
                        source_name=current_qualified_name,
                        target_name=target_name,
                        kind='CALLS'
                    ))
            
            elif node.type == 'object_creation_expression':
                type_node = node.child_by_field_name('type')
                if type_node:
                    target_name = get_identifier_name(type_node)
                    edges.append(SymbolEdge(
                        source_name=current_qualified_name,
                        target_name=target_name,
                        kind='CALLS'
                    ))

            for child in node.children:
                traverse(child, current_qualified_name)

        traverse(tree.root_node, "")
        return edges

    # -------------------------------------------------------------------------
    # TypeScript / JavaScript Parsing
    # -------------------------------------------------------------------------
    def _parse_ts_js_symbols(self, tree: tree_sitter.Tree, content: bytes, language: str) -> list[Symbol]:
        symbols = []
        
        def traverse(node: tree_sitter.Node, parent_qualified_name: str):
            current_qualified_name = parent_qualified_name
            
            if node.type in ('class_declaration', 'interface_declaration', 'enum_declaration'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    kind = node.type.split('_')[0]
                    name = self._get_node_text(name_node, content)
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
                    
                    is_test = name.endswith(('Tests', 'Test', 'TestCase')) or name.startswith(('Test_', 'Test', 'test_'))
                    if is_test:
                        kind = 'test'

                    body_node = node.child_by_field_name('body')
                    if body_node:
                        sig = content[node.start_byte:body_node.start_byte].decode('utf-8', errors='replace').strip()
                    else:
                        sig = content[node.start_byte:node.end_byte].decode('utf-8', errors='replace').strip()

                    symbols.append(Symbol(
                        name=name, qualified_name=current_qualified_name,
                        kind=kind, language=language,
                        line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                        signature=sig, body=self._get_node_text(node, content),
                        is_test=is_test
                    ))
            
            elif node.type in ('function_declaration', 'method_definition'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_node_text(name_node, content)
                    kind = 'function' if node.type == 'function_declaration' else 'method'
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
                    
                    is_test = name.startswith(('test_', 'Test_')) or name.endswith(('Test', 'Tests'))
                    if is_test:
                        kind = 'test'

                    body_node = node.child_by_field_name('body')
                    if body_node:
                        sig = content[node.start_byte:body_node.start_byte].decode('utf-8', errors='replace').strip()
                    else:
                        sig = content[node.start_byte:node.end_byte].decode('utf-8', errors='replace').strip()

                    symbols.append(Symbol(
                        name=name, qualified_name=current_qualified_name,
                        kind=kind, language=language,
                        line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                        signature=sig, body=self._get_node_text(node, content),
                        is_test=is_test
                    ))

            elif node.type == 'call_expression':
                function_node = node.child_by_field_name('function')
                if function_node:
                    fn_name = self._get_node_text(function_node, content)
                    if fn_name in ('describe', 'it', 'test'):
                        # Извлекаем название теста из первого аргумента
                        args_node = node.child_by_field_name('arguments')
                        test_title = None
                        if args_node and args_node.children:
                            for arg in args_node.children:
                                if arg.type in ('string', 'template_string'):
                                    test_title = self._get_node_text(arg, content).strip('\'"` ')
                                    break
                        if test_title:
                            test_qname = f"{parent_qualified_name}.{test_title}" if parent_qualified_name else test_title
                            body = self._get_node_text(node, content)
                            symbols.append(Symbol(
                                name=test_title,
                                qualified_name=test_qname,
                                kind='test',
                                language=language,
                                line_start=node.start_point[0] + 1,
                                line_end=node.end_point[0] + 1,
                                signature=f"{fn_name}('{test_title}')",
                                body=body,
                                is_test=True
                            ))

            elif node.type == 'type_alias_declaration':
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_node_text(name_node, content)
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
                    
                    eq_index = self._get_node_text(node, content).find('=')
                    if eq_index > 0:
                        sig = self._get_node_text(node, content)[:eq_index].strip()
                    else:
                        sig = name

                    symbols.append(Symbol(
                        name=name, qualified_name=current_qualified_name,
                        kind='type', language=language,
                        line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                        signature=sig, body=self._get_node_text(node, content)
                    ))
            
            elif node.type in ('public_field_definition', 'property_signature'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_node_text(name_node, content)
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
                    sig = self._get_node_text(node, content).strip()
                    symbols.append(Symbol(
                        name=name, qualified_name=current_qualified_name,
                        kind='property' if node.type == 'property_signature' else 'field', language=language,
                        line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                        signature=sig, body=self._get_node_text(node, content)
                    ))
            
            elif node.type == 'variable_declarator':
                value_node = node.child_by_field_name('value')
                if value_node and value_node.type == 'arrow_function':
                    name_node = node.child_by_field_name('name')
                    if name_node:
                        name = self._get_node_text(name_node, content)
                        current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
                        is_test = name.startswith(('test_', 'Test_')) or name.endswith(('Test', 'Tests'))
                        body_node = value_node.child_by_field_name('body')
                        if body_node:
                            sig = content[node.start_byte:body_node.start_byte].decode('utf-8', errors='replace').strip()
                        else:
                            sig = name

                        symbols.append(Symbol(
                            name=name, qualified_name=current_qualified_name,
                            kind='test' if is_test else 'function', language=language,
                            line_start=node.start_point[0] + 1, line_end=node.end_point[0] + 1,
                            signature=sig, body=self._get_node_text(node, content),
                            is_test=is_test
                        ))

            for child in node.children:
                traverse(child, current_qualified_name)

        traverse(tree.root_node, "")
        return symbols

    def _extract_ts_js_edges(self, tree: tree_sitter.Tree, content: bytes, language: str) -> list[SymbolEdge]:
        edges = []
        
        def get_identifier_name(node: tree_sitter.Node) -> str:
            if node.type == 'identifier':
                return self._get_node_text(node, content)
            elif node.type == 'member_expression':
                prop_node = node.child_by_field_name('property')
                if prop_node:
                    return self._get_node_text(prop_node, content)
            return self._get_node_text(node, content)

        def traverse(node: tree_sitter.Node, parent_qualified_name: str):
            current_qualified_name = parent_qualified_name
            
            if node.type in ('class_declaration', 'interface_declaration', 'enum_declaration'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_node_text(name_node, content)
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name
                
                for child in node.children:
                    if child.type == 'class_heritage':
                        for heritage_child in child.children:
                            if heritage_child.type == 'extends_clause':
                                for type_node in heritage_child.children:
                                    if type_node.type in ('identifier', 'type_identifier'):
                                        edges.append(SymbolEdge(
                                            source_name=current_qualified_name,
                                            target_name=self._get_node_text(type_node, content),
                                            kind='INHERITS'
                                        ))
                            elif heritage_child.type == 'implements_clause':
                                for type_node in heritage_child.children:
                                    if type_node.type in ('identifier', 'type_identifier'):
                                        edges.append(SymbolEdge(
                                            source_name=current_qualified_name,
                                            target_name=self._get_node_text(type_node, content),
                                            kind='IMPLEMENTS'
                                        ))
                    elif child.type == 'extends_clause':
                        for type_node in child.children:
                            if type_node.type in ('identifier', 'type_identifier'):
                                edges.append(SymbolEdge(
                                    source_name=current_qualified_name,
                                    target_name=self._get_node_text(type_node, content),
                                    kind='INHERITS'
                                ))

            elif node.type in ('function_declaration', 'method_definition'):
                name_node = node.child_by_field_name('name')
                if name_node:
                    name = self._get_node_text(name_node, content)
                    current_qualified_name = f"{parent_qualified_name}.{name}" if parent_qualified_name else name

            # Детекция TESTED_BY для call_expression describe/it/test
            if node.type == 'call_expression':
                function_node = node.child_by_field_name('function')
                if function_node:
                    fn_name = self._get_node_text(function_node, content)
                    if fn_name in ('describe', 'it', 'test'):
                        args_node = node.child_by_field_name('arguments')
                        test_title = None
                        if args_node and args_node.children:
                            for arg in args_node.children:
                                if arg.type in ('string', 'template_string'):
                                    test_title = self._get_node_text(arg, content).strip('\'"` ')
                                    break
                        if test_title:
                            test_qname = f"{parent_qualified_name}.{test_title}" if parent_qualified_name else test_title
                            target_candidate = extract_tested_target(test_title)
                            if target_candidate and target_candidate != test_title:
                                edges.append(SymbolEdge(
                                    source_name=target_candidate,
                                    target_name=test_qname,
                                    kind='TESTED_BY'
                                ))

            # Edge Extraction: imports, calls, new
            if node.type == 'import_statement':
                for child in node.children:
                    if child.type == 'import_clause':
                        for c in child.children:
                            if c.type == 'identifier':
                                edges.append(SymbolEdge(
                                    source_name=current_qualified_name or "file_level",
                                    target_name=self._get_node_text(c, content),
                                    kind='IMPORTS'
                                ))
                            elif c.type == 'named_imports':
                                for spec in c.children:
                                    if spec.type == 'import_specifier':
                                        name_node = spec.child_by_field_name('name')
                                        if name_node:
                                            edges.append(SymbolEdge(
                                                source_name=current_qualified_name or "file_level",
                                                target_name=self._get_node_text(name_node, content),
                                                kind='IMPORTS'
                                            ))
            elif node.type == 'call_expression':
                function_node = node.child_by_field_name('function')
                if function_node:
                    target_name = get_identifier_name(function_node)
                    if target_name not in ('describe', 'it', 'test'):
                        edges.append(SymbolEdge(
                            source_name=current_qualified_name,
                            target_name=target_name,
                            kind='CALLS'
                        ))
            elif node.type == 'new_expression':
                constructor_node = node.child_by_field_name('constructor')
                if constructor_node:
                    target_name = get_identifier_name(constructor_node)
                    edges.append(SymbolEdge(
                        source_name=current_qualified_name,
                        target_name=target_name,
                        kind='CALLS'
                    ))

            for child in node.children:
                traverse(child, current_qualified_name)

        traverse(tree.root_node, "")
        return edges

    # -------------------------------------------------------------------------
    # SQL Parsing
    # -------------------------------------------------------------------------
    def _parse_sql_symbols(self, tree: tree_sitter.Tree, content: bytes) -> list[Symbol]:
        symbols = []
        
        def traverse(node: tree_sitter.Node):
            if node.type in ('create_table', 'create_view', 'create_index', 'create_function'):
                name = None
                for child in node.children:
                    if child.type in ('identifier', 'object_reference', 'table_reference'):
                        name = self._get_node_text(child, content)
                        break
                
                if name:
                    kind = node.type.split('_')[1]
                    body_node = node
                    parent = node.parent
                    if parent and parent.type == 'statement':
                        body_node = parent
                        
                    symbols.append(Symbol(
                        name=name, qualified_name=name,
                        kind=kind, language='sql',
                        line_start=body_node.start_point[0] + 1, line_end=body_node.end_point[0] + 1,
                        signature=name, body=self._get_node_text(body_node, content)
                    ))

            for child in node.children:
                traverse(child)

        traverse(tree.root_node)
        return symbols

    def _extract_sql_edges(self, tree: tree_sitter.Tree, content: bytes) -> list[SymbolEdge]:
        return []
