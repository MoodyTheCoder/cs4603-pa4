
import ast
import operator as op
from unitycatalog.ai.core.databricks import DatabricksFunctionClient

from dotenv import load_dotenv
load_dotenv()   

# ---- safe arithmetic evaluator (from tools/mcp_server.py) ----
_BIN_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
    ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
    ast.FloorDiv: op.floordiv,
}
_UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}

def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Expression contains an unsupported operation")

# ---- the tools ----
def compound_growth(principal: float, rate: float, periods: int) -> float:
    """Return principal * (1 + rate) ** periods.

    Args:
        principal: starting value.
        rate: growth rate per period as a decimal (0.08 = 8%).
        periods: number of periods.
    """
    return principal * (1 + rate) ** periods

def percent_change(old_value: float, new_value: float) -> float:
    """Compute percentage change from old_value to new_value.

    Args:
        old_value: the baseline value
        new_value: the new value

    Returns:
        The percentage change (positive = increase, negative = decrease)
    """
    if old_value == 0:
        return None
    return ((new_value - old_value) / abs(old_value)) * 100

def calculate(expression: str) -> float:
    """Evaluate a math expression safely. Supports + - * / ** % and parentheses.

    Args:
        expression: A string like '16.91 * (1.08 ** 3)'

    Returns:
        The numeric result of the expression.
    """
    import ast
    import operator as op

    _BIN_OPS = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.Mod: op.mod,
        ast.FloorDiv: op.floordiv,
    }
    _UNARY_OPS = {ast.UAdd: op.pos, ast.USub: op.neg}

    def _safe_eval(node):
        if isinstance(node, ast.Expression):
            return _safe_eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return float(node.value)
            raise ValueError(f"Unsupported constant: {node.value!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
        raise ValueError("Expression contains an unsupported operation")

    tree = ast.parse(expression, mode="eval")
    return _safe_eval(tree)


# ---- register them ----
client = DatabricksFunctionClient()
for func in [compound_growth, percent_change, calculate]:
    client.create_python_function(
        func=func,
        catalog="main",
        schema="default",
        replace=True,
    )
    print(f"✅ Registered main.default.{func.__name__}")