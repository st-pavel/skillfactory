"""
Модуль для анализа математических функций.

Предоставляет класс FunctionAnalyzer для всестороннего анализа функций:
- Построение графиков функции и её производных
- Определение области определения и значений
- Поиск асимптот всех типов
- Анализ особых точек
- Проверка на четность/нечетность
- Анализ выпуклости

Пример использования:
    from functionanalyzer import FunctionAnalyzer
    from sympy import Symbol
    
    x = Symbol('x')
    f = (x**3)/(x**2+x-1)
    
    analyzer = FunctionAnalyzer(f, x)
    analyzer.describe()  # выводит подробный анализ
    analyzer.plot()      # строит график
"""

import numpy as np
from sympy import (
    Symbol, lambdify, diff, S, solve, limit, oo,
    fraction, sympify, zoo, log, sqrt, tan, Pow,
    simplify, cos, sin, Eq, solveset, Interval,
    exp  # добавили exp для экспоненциальных функций
)
from sympy.calculus.util import (
    function_range,
    continuous_domain
)

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import Markdown, display

import warnings

__all__ = ['FunctionAnalyzer']

warnings.filterwarnings('ignore')


class FunctionAnalyzer:
    def __init__(self, f, x, x_range=(-10, 10), y_range=(-10, 10), points=300):
        """
        Инициализация анализатора функций
        
        Args:
            f: символьное выражение функции
            x: символьная переменная
            x_range: диапазон по x для построения графика
            y_range: диапазон по y для построения графика
            points: количество точек для построения
        """
        self.f = f
        self.x = x
        self.x_range = x_range
        self.y_range = y_range
        self.points = points
        
        # Вычисляем производные
        self.f_prime = diff(f, x)
        self.f_double_prime = diff(self.f_prime, x)
        
        # Находим область определения
        self.domain = self._find_domain()
        
        # Находим особые точки
        self.special_points = self._find_special_points()
        
        # Находим асимптоты
        self.vertical_asymptotes, self.horizontal_asymptotes, self.oblique_asymptotes = self._find_asymptotes()

    def _find_domain(self):
        """Определяет область определения функции"""
        try:
            domain = continuous_domain(self.f, self.x, S.Reals)
            
            # Дополнительные проверки для разных типов функций
            if any(isinstance(arg, log) for arg in self.f.atoms(log)):
                # Для логарифмов проверяем положительность подлогарифмического выражения
                for log_term in self.f.atoms(log):
                    domain = domain & solve(log_term.args[0] > 0, self.x)
                    
            if any(isinstance(arg, Pow) and arg.exp.is_rational and arg.exp.q == 2 for arg in self.f.atoms()):
                # Для корней проверяем неотрицательность подкоренного выражения
                for sqrt_term in self.f.atoms(sqrt):
                    domain = domain & solve(sqrt_term.args[0] >= 0, self.x)
                    
            if any(isinstance(arg, tan) for arg in self.f.atoms(tan)):
                # Для тангенса исключаем точки π/2 + πn
                domain = domain & solve(cos(self.x) != 0, self.x)
                
            return domain
        except Exception as e:
            print(f"Предупреждение при определении области определения: {str(e)}")
            return S.Reals

    def _find_special_points(self):
        """Находит особые точки функции"""
        special_points = []
        try:
            # Точки разрыва
            if hasattr(self.domain, 'args'):
                for interval in self.domain.args:
                    if interval.left.is_real:
                        special_points.append(float(interval.left))
                    if interval.right.is_real:
                        special_points.append(float(interval.right))
            
            # Точки экстремума (производная равна 0)
            critical_points = solve(self.f_prime, self.x)
            special_points.extend([float(p) for p in critical_points if p.is_real])
            
            # Точки перегиба (вторая производная равна 0)
            inflection_points = solve(self.f_double_prime, self.x)
            special_points.extend([float(p) for p in inflection_points if p.is_real])
            
            return sorted(list(set(special_points)))
        except Exception as e:
            print(f"Предупреждение при поиске особых точек: {str(e)}")
            return []

    def _find_asymptotes(self):
        """Находит все типы асимптот"""
        vertical = []
        horizontal = []
        oblique = []
        
        try:
            # Вертикальные асимптоты
            for point in self.special_points:
                try:
                    lim_left = limit(self.f, self.x, point, dir='-')
                    lim_right = limit(self.f, self.x, point, dir='+')
                    if lim_left.is_infinite or lim_right.is_infinite:
                        vertical.append(point)
                except:
                    continue
            
            # Горизонтальные и наклонные асимптоты
            try:
                lim_plus_inf = limit(self.f, self.x, oo)
                lim_minus_inf = limit(self.f, self.x, -oo)
                
                if lim_plus_inf.is_real:
                    horizontal.append(float(lim_plus_inf))
                elif lim_plus_inf is zoo:
                    # Проверяем на наклонную асимптоту
                    k = limit(self.f/self.x, self.x, oo)
                    if k.is_real:
                        b = limit(self.f - k*self.x, self.x, oo)
                        if b.is_real:
                            oblique.append((float(k), float(b)))  # y = kx + b
                
                if lim_minus_inf.is_real and lim_minus_inf != lim_plus_inf:
                    horizontal.append(float(lim_minus_inf))
                
            except:
                pass
                
            return vertical, horizontal, oblique
        except Exception as e:
            print(f"Предупреждение при поиске асимптот: {str(e)}")
            return [], [], []

    def _safe_eval(self, numpy_func, x_vals):
        """Безопасное вычисление значений функции"""
        try:
            result = numpy_func(x_vals)
            # Заменяем бесконечности и nan на None для plotly
            result[~np.isfinite(result)] = None
            return result
        except:
            return np.full_like(x_vals, None, dtype=float)

    def plot(self):
        """Создает интерактивный график с помощью plotly"""
        fig = go.Figure()

        # Создаем массив точек с учетом особых точек
        x_points = np.array(sorted(list(set(self.special_points + 
                                          [self.x_range[0], self.x_range[1]]))))
        
        # Добавляем точки между особыми точками
        x_arrays = []
        for i in range(len(x_points) - 1):
            if x_points[i+1] - x_points[i] > 1e-10:  # Избегаем слишком близких точек
                x_arrays.append(np.linspace(x_points[i] + 1e-10, 
                                          x_points[i+1] - 1e-10, 
                                          self.points))

        # Преобразуем функции в numpy
        f_numpy = lambdify(self.x, self.f, 'numpy')
        f_prime_numpy = lambdify(self.x, self.f_prime, 'numpy')
        f_double_prime_numpy = lambdify(self.x, self.f_double_prime, 'numpy')

        # Строим графики для каждого интервала
        for x_arr in x_arrays:
            y = self._safe_eval(f_numpy, x_arr)
            y_prime = self._safe_eval(f_prime_numpy, x_arr)
            y_double = self._safe_eval(f_double_prime_numpy, x_arr)

            fig.add_trace(go.Scatter(x=x_arr, y=y, name='f(x)', 
                                   line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=x_arr, y=y_prime, name="f'(x)", 
                                   line=dict(color='green')))
            fig.add_trace(go.Scatter(x=x_arr, y=y_double, name='f"(x)', 
                                   line=dict(color='red')))

        # Добавляем асимптоты
        for va in self.vertical_asymptotes:
            fig.add_trace(go.Scatter(x=[va, va], 
                                   y=[self.y_range[0], self.y_range[1]],
                                   mode='lines', 
                                   line=dict(color='gray', dash='dash'),
                                   name=f'x = {va}'))

        for ha in self.horizontal_asymptotes:
            fig.add_trace(go.Scatter(x=[self.x_range[0], self.x_range[1]], 
                                   y=[ha, ha],
                                   mode='lines', 
                                   line=dict(color='red', dash='dot'),
                                   name=f'y = {ha}'))

        for k, b in self.oblique_asymptotes:
            x_oblique = np.array([self.x_range[0], self.x_range[1]])
            y_oblique = k * x_oblique + b
            fig.add_trace(go.Scatter(x=x_oblique, y=y_oblique,
                                   mode='lines',
                                   line=dict(color='purple', dash='dot'),
                                   name=f'y = {k}x + {b}'))

        # Настройка внешнего вида
        fig.update_layout(
            title='График функции и её производных',
            xaxis_title='x',
            yaxis_title='y',
            showlegend=True,
            hovermode='x unified',
            plot_bgcolor='white',
            xaxis=dict(
                range=self.x_range,
                gridcolor='lightgray',
                zerolinecolor='black',
                zerolinewidth=0.5,
            ),
            yaxis=dict(
                range=self.y_range,
                gridcolor='lightgray',
                zerolinecolor='black',
                zerolinewidth=0.5,
            ),
        )

        fig.show()
        
    def describe(self):
        """Выводит подробный анализ функции в формате Markdown"""
        from sympy import latex, Eq, solveset, Interval
        from IPython.display import display, Markdown
        
        # Проверка на четность/нечетность
        f_minus_x = self.f.subs(self.x, -self.x)
        is_even = simplify(f_minus_x - self.f) == 0
        is_odd = simplify(f_minus_x + self.f) == 0
        
        # Находим точки пересечения с осями
        x_intersections = solveset(Eq(self.f, 0), self.x)
        y_intersection = self.f.subs(self.x, 0)
        
        f_prime_intersections = solveset(Eq(self.f_prime, 0), self.x)
        f_prime_y_intersection = self.f_prime.subs(self.x, 0)
        
        f_double_prime_intersections = solveset(Eq(self.f_double_prime, 0), self.x)
        f_double_prime_y_intersection = self.f_double_prime.subs(self.x, 0)
        
        # Проверка выпуклости
        try:
            range_second_derivative = function_range(self.f_double_prime, self.x, S.Reals)
            is_concave_up = range_second_derivative.is_subset(Interval(0, oo))
            is_concave_down = range_second_derivative.is_subset(Interval(-oo, 0))
        except:
            is_concave_up = "Не удалось определить"
            is_concave_down = "Не удалось определить"
            
        # Безопасное получение области значений
        def safe_range(func):
            try:
                return latex(function_range(func, self.x, S.Reals))
            except:
                return r"\text{не удалось определить}"
        
        description = f"""
### Анализ функции
#### Функция:
* $f(x) = {latex(self.f)}$ 

* $ D(f(x)) : {latex(self.domain)}$
    
* $ E(f(x)) : {safe_range(self.f)}$
    
* Вертикальные асимптоты:   $x = {latex(self.vertical_asymptotes)}$
    
* Горизонтальные асимптоты: $y = {latex(self.horizontal_asymptotes)}$
    
* Наклонные асимптоты: {self._format_oblique_asymptotes()}
    
* Точки пересечения с осью x: ${latex(x_intersections)}$
    
* Точка пересечения с осью y: ${latex(y_intersection)}$

#### Производная функции:

* $f'(x) = {latex(self.f_prime)}$

* $ D(f'(x)) : {latex(continuous_domain(self.f_prime, self.x, S.Reals))}$
    
* $ E(f'(x)) : {safe_range(self.f_prime)}$

* Точки пересечения с осью x: ${latex(f_prime_intersections)}$
    
* Точка пересечения с осью y: ${latex(f_prime_y_intersection)}$

#### Вторая производная функции:

* $f''(x) = {latex(self.f_double_prime)}$

* $ D(f''(x)) : {latex(continuous_domain(self.f_double_prime, self.x, S.Reals))}$
    
* $ E(f''(x)) : {safe_range(self.f_double_prime)}$
    
* Точки пересечения с осью x: ${latex(f_double_prime_intersections)}$
    
* Точка пересечения с осью y: ${latex(f_double_prime_y_intersection)}$

#### Проверка функции на четность:

* $f(-x) = {latex(f_minus_x)}$
    
* $f(x) = {latex(self.f)}$

* Функция {'четная' if is_even else 'нечетная' if is_odd else 'общего вида'}.
    
#### Проверка функции на выпуклость:

* Функция выпуклая вверх: {is_concave_down}
    
* Функция выпуклая вниз: {is_concave_up}

#### Особые точки функции:
* {self._format_special_points()}
"""
        display(Markdown(description))

    def _format_oblique_asymptotes(self):
        """Форматирует наклонные асимптоты для вывода в LaTeX"""
        if not self.oblique_asymptotes:
            return "отсутствуют"
        
        asymptotes = []
        for k, b in self.oblique_asymptotes:
            if b >= 0:
                asymptotes.append(f"y = {k}x + {b}")
            else:
                asymptotes.append(f"y = {k}x - {abs(b)}")
        
        return "$" + ",\\ ".join(asymptotes) + "$"

    def _format_special_points(self):
        """Форматирует особые точки для вывода"""
        if not self.special_points:
            return "особых точек не найдено"
        
        points = []
        for point in self.special_points:
            types = []
            if point in self.vertical_asymptotes:
                types.append("точка разрыва")
            try:
                if self.f_prime.subs(self.x, point) == 0:
                    types.append("критическая точка")
            except:
                pass
            try:
                if self.f_double_prime.subs(self.x, point) == 0:
                    types.append("точка перегиба")
            except:
                pass
            
            points.append(f"x = {point} ({', '.join(types)})")
        
        return "\n   * ".join(points)
