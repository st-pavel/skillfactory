"""
FunctionAnalyzer package for mathematical function analysis.
"""

from sympy import (
    Symbol, sqrt, log, sin, cos, exp, tan
)
from .functionanalyzer import FunctionAnalyzer

# Экспортируем все необходимые символы
__all__ = [
    'FunctionAnalyzer',
    'Symbol',
    'sqrt',
    'log',
    'sin',
    'cos',
    'exp',
    'tan'
]

# Делаем все импортированные символы доступными на уровне пакета
Symbol = Symbol
sqrt = sqrt
log = log
sin = sin
cos = cos
exp = exp
tan = tan

__version__ = '1.0.0'
__author__ = 'Pavel Stepurin'