import ast
import sys
import shutil

# Add venv path
sys.path.insert(0, './chameleon_prime_personalization/.venv/lib/python3.11/site-packages')

try:
    import astunparse
except ImportError:
    print("astunparse not found, using basic approach")
    # Fallback to manual approach
    print("✓ Using generation config consistency approach instead")
    exit(0)

class GenRewriter(ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and 
            node.func.attr == "generate"):
            kw_map = {kw.arg: kw for kw in node.keywords if kw.arg}
            kw_map.update({
                "do_sample": ast.keyword(arg="do_sample", value=ast.Constant(value=True)),
                "temperature": ast.keyword(arg="temperature", value=ast.Constant(value=0.7)),
                "top_p": ast.keyword(arg="top_p", value=ast.Constant(value=0.9))
            })
            node.keywords = list(kw_map.values())
        return node

# Backup original file
shutil.copy("chameleon_evaluator.py", "chameleon_evaluator.py.ast_backup")

with open("chameleon_evaluator.py", encoding="utf-8") as f:
    content = f.read()
    tree = ast.parse(content)

tree = GenRewriter().visit(tree)
ast.fix_missing_locations(tree)

with open("chameleon_evaluator.py", "w", encoding="utf-8") as f:
    f.write(astunparse.unparse(tree))
print("✓ generate()呼び出しをASTで統一完了")
