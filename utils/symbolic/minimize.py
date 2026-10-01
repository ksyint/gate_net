"""Exact Boolean feature expressions with optional Espresso two-level minimization."""
from functools import lru_cache


def minimize_features(circuit, max_support=16):
    from pyeda.inter import expr, exprvar, espresso_exprs
    if circuit.get('format') != 'oslgn-circuit-v1':
        raise ValueError('Unsupported circuit format')
    if max_support < 1:
        raise ValueError('max_support must be positive')

    @lru_cache(None)
    def node(depth, index):
        if depth < 0:
            return exprvar('x', index)
        layer = circuit['layers'][depth]
        gate = layer['gate'][index]
        if gate in (0, 15):
            return expr(int(gate == 15))
        a, b = node(depth - 1, layer['left'][index]), node(depth - 1, layer['right'][index])
        result = expr(0)
        for av, bv in ((0, 0), (0, 1), (1, 0), (1, 1)):
            if (gate >> (3 - (2 * av + bv))) & 1:
                result |= (a if av else ~a) & (b if bv else ~b)
        return result.simplify()

    outputs = []
    last = len(circuit['layers']) - 1
    for index in range(len(circuit['layers'][last]['gate'])):
        expression = node(last, index)
        support = len(expression.support)
        eligible = support <= max_support and not expression.is_zero() and not expression.is_one()
        minimized = espresso_exprs(expression.to_dnf())[0] if eligible else expression
        outputs.append({'feature': index, 'support_variables': support,
                        'espresso_applied': eligible, 'expression': str(expression),
                        'result': str(minimized)})
    return outputs
