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
    exp, latex, FiniteSet, Union  # добавили Union
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
    def __init__(self, f, x, x_range=(-10, 10), y_range=(-10, 10), points=300, complex_domain=False):
        """
        Инициализация анализатора функций
        
        Args:
            f: символьное выражение функции
            x: символьная переменная
            x_range: диапазон по x для построения графика
            y_range: диапазон по y для построения графика
            points: количество точек для построения
            complex_domain: если True, ищет решения в комплексных числах
        """
        # Упрощаем исходную функцию
        self.f = simplify(f)
        self.x = x
        self.x_range = x_range
        self.y_range = y_range
        self.points = points
        self.complex_domain = complex_domain
        
        # Вычисляем и упрощаем производные
        self.f_prime = simplify(diff(self.f, x))
        self.f_double_prime = simplify(diff(self.f_prime, x))
        
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
                    # Преобразуем решение в Interval
                    condition = log_term.args[0] > 0
                    solution = solveset(condition, self.x, domain=S.Reals)
                    if isinstance(solution, Union):
                        domain = domain & solution
                    else:
                        domain = domain & solution
            
            if any(isinstance(arg, Pow) and arg.is_Pow and arg.exp == S.Half for arg in self.f.atoms()):
                # Для корней проверяем неотрицательность подкоренного выражения
                for sqrt_term in self.f.atoms(sqrt):
                    condition = sqrt_term.args[0] >= 0
                    solution = solveset(condition, self.x, domain=S.Reals)
                    if isinstance(solution, Union):
                        domain = domain & solution
                    else:
                        domain = domain & solution
            
            if any(isinstance(arg, tan) for arg in self.f.atoms(tan)):
                # Для тангенса исключаем точки π/2 + πn
                condition = cos(self.x) != 0
                solution = solveset(condition, self.x, domain=S.Reals)
                if isinstance(solution, Union):
                    domain = domain & solution
                else:
                    domain = domain & solution
            
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
            if self.complex_domain:
                critical_points = solve(self.f_prime, self.x)  # Ищет все корни, включая комплексные
            else:
                critical_points = solve(self.f_prime, self.x, domain=S.Reals)  # Только действительные корни
            special_points.extend([complex(p) for p in critical_points])
            
            # Точки перегиба (вторая производная равна 0)
            if self.complex_domain:
                inflection_points = solve(self.f_double_prime, self.x)
            else:
                inflection_points = solve(self.f_double_prime, self.x, domain=S.Reals)
            special_points.extend([complex(p) for p in inflection_points])
            
            # Если работаем только с действительными числами, фильтруем комплексные
            if not self.complex_domain:
                special_points = [p for p in special_points if not p.imag]
                special_points = [float(p.real) for p in special_points]
            
            return sorted(list(set(special_points)), key=lambda x: (x.real, x.imag) if isinstance(x, complex) else (x, 0))
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
                # Проверяем сначала на наклонную асимптоту
                k = limit(self.f/self.x, self.x, oo)
                
                if k.is_real and k != 0:  # Если k вещественное и не ноль - есть наклонная асимптота
                    b = limit(self.f - k*self.x, self.x, oo)
                    if b.is_real:
                        oblique.append((float(k), float(b)))  # y = kx + b
                else:  # Иначе проверяем на горизонтальную асимптоту
                    lim_plus_inf = limit(self.f, self.x, oo)
                    lim_minus_inf = limit(self.f, self.x, -oo)
                    
                    if lim_plus_inf.is_real:
                        horizontal.append(float(lim_plus_inf))
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
            # Если результат - скаляр (константа), преобразуем его в массив
            if np.isscalar(result):
                result = np.full_like(x_vals, result)
            # Заменяем бесконечности и nan на None для plotly
            result[~np.isfinite(result)] = None
            return result
        except:
            return np.full_like(x_vals, None, dtype=float)

    def plot(self):
        """Создает интерактивный график с помощью plotly"""
        fig = go.Figure()

        # Создаем массив точек с учетом особых точек
        # Для комплексных чисел берем только действительную часть
        real_special_points = []
        for point in self.special_points:
            if isinstance(point, complex):
                if abs(point.imag) < 1e-10:  # Если мнимая часть близка к нулю
                    real_special_points.append(float(point.real))
            else:
                real_special_points.append(float(point))
                
        x_points = np.array(sorted(list(set(real_special_points + 
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

            # Для комплексных значений берем только действительную часть
            if self.complex_domain:
                if y is not None and not isinstance(y, type(None)):
                    y = np.real(y)
                if y_prime is not None and not isinstance(y_prime, type(None)):
                    y_prime = np.real(y_prime)
                if y_double is not None and not isinstance(y_double, type(None)):
                    y_double = np.real(y_double)

            fig.add_trace(go.Scatter(x=x_arr, y=y, name='f(x)', 
                                   line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=x_arr, y=y_prime, name="f'(x)", 
                                   line=dict(color='green')))
            fig.add_trace(go.Scatter(x=x_arr, y=y_double, name='f"(x)', 
                                   line=dict(color='red')))

        # Добавляем асимптоты
        for va in self.vertical_asymptotes:
            if isinstance(va, complex):
                if abs(va.imag) < 1e-10:
                    va = float(va.real)
                else:
                    continue
            fig.add_trace(go.Scatter(x=[va, va], 
                                   y=[self.y_range[0], self.y_range[1]],
                                   mode='lines', 
                                   line=dict(color='gray', dash='dash'),
                                   name=f'x = {va}'))

        for ha in self.horizontal_asymptotes:
            if isinstance(ha, complex):
                if abs(ha.imag) < 1e-10:
                    ha = float(ha.real)
                else:
                    continue
            fig.add_trace(go.Scatter(x=[self.x_range[0], self.x_range[1]], 
                                   y=[ha, ha],
                                   mode='lines', 
                                   line=dict(color='red', dash='dot'),
                                   name=f'y = {ha}'))

        for k, b in self.oblique_asymptotes:
            if isinstance(k, complex) or isinstance(b, complex):
                if abs(k.imag) < 1e-10 and abs(b.imag) < 1e-10:
                    k = float(k.real)
                    b = float(b.real)
                else:
                    continue
            x_oblique = np.array([self.x_range[0], self.x_range[1]])
            y_oblique = k * x_oblique + b
            fig.add_trace(go.Scatter(x=x_oblique, y=y_oblique,
                                   mode='lines',
                                   line=dict(color='purple', dash='dot'),
                                   name=f'y = {k}x + {b}'))

        # Настройка внешнего вида
        title = 'График функции и её производных'
        if self.complex_domain:
            title += ' (действительная часть)'
            
        fig.update_layout(
            title=title,
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
        
    def _analyze_monotonicity(self):
        """Анализирует промежутки возрастания и убывания функции"""
        try:
            # Получаем точки, где производная равна нулю
            critical_points = []
            solutions = solveset(Eq(self.f_prime, 0), self.x, domain=S.Reals)
            if isinstance(solutions, FiniteSet):
                critical_points.extend([float(p) for p in solutions if p.is_real])
            
            # Добавляем точки разрыва из области определения
            domain_points = []
            if isinstance(self.domain, Union):
                intervals = self.domain.args
            elif isinstance(self.domain, Interval):
                intervals = [self.domain]
            else:
                intervals = []
                
            for interval in intervals:
                if interval.left != -oo and interval.left.is_real:
                    domain_points.append(float(interval.left))
                if interval.right != oo and interval.right.is_real:
                    domain_points.append(float(interval.right))
            
            # Объединяем и сортируем все точки
            all_points = sorted(set(critical_points + domain_points))
            
            if not all_points:
                # Если нет критических точек, проверяем знак производной в одной точке
                test_point = 0
                try:
                    sign = float(self.f_prime.subs(self.x, test_point).evalf())
                    if sign > 0:
                        return "функция возрастает на всей области определения"
                    elif sign < 0:
                        return "функция убывает на всей области определения"
                    else:
                        return "функция постоянна"
                except:
                    return "не удалось определить монотонность в точке 0"
            
            # Анализируем знак производной в каждом интервале
            intervals = []
            
            # Обрабатываем каждый интервал области определения
            if isinstance(self.domain, Union):
                domain_intervals = self.domain.args
            elif isinstance(self.domain, Interval):
                domain_intervals = [self.domain]
            else:
                domain_intervals = []
            
            for domain_interval in domain_intervals:
                left = domain_interval.left
                right = domain_interval.right
                
                # Фильтруем точки для текущего интервала
                interval_points = [p for p in all_points 
                                if (left == -oo or p > left) and (right == oo or p < right)]
                interval_points = [left] + interval_points + [right]
                
                for i in range(len(interval_points) - 1):
                    # Выбираем тестовую точку в середине интервала
                    if interval_points[i] == -oo:
                        test_point = float(interval_points[i + 1]) - 1
                    elif interval_points[i + 1] == oo:
                        test_point = float(interval_points[i]) + 1
                    else:
                        test_point = (float(interval_points[i]) + float(interval_points[i + 1])) / 2
                    
                    try:
                        sign = float(self.f_prime.subs(self.x, test_point).evalf())
                        if sign > 0:
                            intervals.append((interval_points[i], interval_points[i + 1], "возрастает"))
                        elif sign < 0:
                            intervals.append((interval_points[i], interval_points[i + 1], "убывает"))
                    except:
                        continue
            
            # Форматируем результат
            result = []
            for left, right, behavior in intervals:
                left_str = '-\\infty' if left == -oo else latex(left)
                right_str = '\\infty' if right == oo else latex(right)
                result.append(f"на интервале $({left_str}; {right_str})$ функция {behavior}")
            
            return "\n   * ".join(result) if result else "не удалось определить промежутки монотонности"
            
        except Exception as e:
            return f"не удалось определить промежутки монотонности: {str(e)}"

    def _analyze_concavity(self):
        """Анализирует промежутки выпуклости функции"""
        try:
            # Получаем точки, где вторая производная равна нулю
            inflection_points = []
            solutions = solveset(Eq(self.f_double_prime, 0), self.x, domain=S.Reals)
            if isinstance(solutions, FiniteSet):
                inflection_points.extend([float(p) for p in solutions if p.is_real])
            
            # Добавляем точки разрыва из области определения
            domain_points = []
            if isinstance(self.domain, Union):
                intervals = self.domain.args
            elif isinstance(self.domain, Interval):
                intervals = [self.domain]
            else:
                intervals = []
                
            for interval in intervals:
                if interval.left != -oo and interval.left.is_real:
                    domain_points.append(float(interval.left))
                if interval.right != oo and interval.right.is_real:
                    domain_points.append(float(interval.right))
            
            # Объединяем и сортируем все точки
            all_points = sorted(set(inflection_points + domain_points))
            
            if not all_points:
                # Если нет точек перегиба, проверяем знак второй производной в одной точке
                test_point = 0
                try:
                    sign = float(self.f_double_prime.subs(self.x, test_point).evalf())
                    if sign > 0:
                        return "функция выпукла вниз на всей области определения"
                    elif sign < 0:
                        return "функция выпукла вверх на всей области определения"
                    else:
                        return "функция линейна"
                except:
                    return "не удалось определить выпуклость в точке 0"
            
            # Анализируем знак второй производной в каждом интервале
            intervals = []
            
            # Обрабатываем каждый интервал области определения
            if isinstance(self.domain, Union):
                domain_intervals = self.domain.args
            elif isinstance(self.domain, Interval):
                domain_intervals = [self.domain]
            else:
                domain_intervals = []
            
            for domain_interval in domain_intervals:
                left = domain_interval.left
                right = domain_interval.right
                
                # Фильтруем точки для текущего интервала
                interval_points = [p for p in all_points 
                                if (left == -oo or p > left) and (right == oo or p < right)]
                interval_points = [left] + interval_points + [right]
                
                for i in range(len(interval_points) - 1):
                    # Выбираем тестовую точку в середине интервала
                    if interval_points[i] == -oo:
                        test_point = float(interval_points[i + 1]) - 1
                    elif interval_points[i + 1] == oo:
                        test_point = float(interval_points[i]) + 1
                    else:
                        test_point = (float(interval_points[i]) + float(interval_points[i + 1])) / 2
                    
                    try:
                        sign = float(self.f_double_prime.subs(self.x, test_point).evalf())
                        if sign > 0:
                            intervals.append((interval_points[i], interval_points[i + 1], "выпукла вниз"))
                        elif sign < 0:
                            intervals.append((interval_points[i], interval_points[i + 1], "выпукла вверх"))
                    except:
                        continue
            
            # Форматируем результат
            result = []
            for left, right, behavior in intervals:
                left_str = '-\\infty' if left == -oo else latex(left)
                right_str = '\\infty' if right == oo else latex(right)
                result.append(f"на интервале $({left_str}; {right_str})$ функция {behavior}")
            
            return "\n   * ".join(result) if result else "не удалось определить промежутки выпуклости"
            
        except Exception as e:
            return f"не удалось определить промежутки выпуклости: {str(e)}"

    def _format_asymptotes(self):
        """Форматирует уравнения асимптот для вывода"""
        result = []
        
        # Вертикальные асимптоты
        if self.vertical_asymptotes:
            for x in self.vertical_asymptotes:
                result.append(f"* Вертикальная асимптота: $x = {latex(x)}$")
        
        # Горизонтальные асимптоты
        if self.horizontal_asymptotes:
            for y in self.horizontal_asymptotes:
                result.append(f"* Горизонтальная асимптота: $y = {latex(y)}$")
        
        # Наклонные асимптоты
        if self.oblique_asymptotes:
            for k, b in self.oblique_asymptotes:
                if b >= 0:
                    result.append(f"* Наклонная асимптота: $y = {latex(k)}x + {latex(b)}$")
                else:
                    result.append(f"* Наклонная асимптота: $y = {latex(k)}x - {latex(abs(b))}$")
        
        return "\n".join(result) if result else "* асимптот нет"

    def _analyze_extrema(self):
        """Анализирует точки экстремума функции"""
        try:
            # Находим точки, где производная равна нулю
            critical_points = []
            solutions = solveset(Eq(self.f_prime, 0), self.x, domain=S.Reals)
            if isinstance(solutions, FiniteSet):
                critical_points.extend([float(p) for p in solutions if p.is_real])
            
            if not critical_points:
                return "точек экстремума нет"
            
            # Определяем тип экстремума для каждой точки
            extrema = []
            for point in critical_points:
                try:
                    # Проверяем знак второй производной
                    second_deriv = float(self.f_double_prime.subs(self.x, point).evalf())
                    if second_deriv > 0:
                        extrema.append((point, "минимум"))
                    elif second_deriv < 0:
                        extrema.append((point, "максимум"))
                    else:
                        # Если вторая производная равна 0, нужен дополнительный анализ
                        extrema.append((point, "точка перегиба или требует дополнительного анализа"))
                except:
                    continue
            
            # Форматируем результат
            result = []
            for point, type_extremum in sorted(extrema):
                value = float(self.f.subs(self.x, point).evalf())
                result.append(f"точка $x = {latex(point)}$ - точка {type_extremum}, значение функции $f({latex(point)}) = {latex(value)}$")
            
            return "\n   * ".join(result) if result else "не удалось определить тип экстремумов"
            
        except Exception as e:
            return f"не удалось определить точки экстремума: {str(e)}"

    def describe(self):
        """Выводит подробный анализ функции в формате Markdown"""
        from sympy import latex, Eq, solveset, Interval
        from IPython.display import display, Markdown
        
        # Проверка на четность/нечетность
        f_minus_x = self.f.subs(self.x, -self.x)
        is_even = simplify(f_minus_x - self.f) == 0
        is_odd = simplify(f_minus_x + self.f) == 0
        
        # Находим точки пересечения с осями
        if self.complex_domain:
            x_intersections = solveset(Eq(self.f, 0), self.x)
            f_prime_intersections = solveset(Eq(self.f_prime, 0), self.x)
            f_double_prime_intersections = solveset(Eq(self.f_double_prime, 0), self.x)
        else:
            x_intersections = solveset(Eq(self.f, 0), self.x, domain=S.Reals)
            f_prime_intersections = solveset(Eq(self.f_prime, 0), self.x, domain=S.Reals)
            f_double_prime_intersections = solveset(Eq(self.f_double_prime, 0), self.x, domain=S.Reals)
            
        y_intersection = self.f.subs(self.x, 0)
        f_prime_y_intersection = self.f_prime.subs(self.x, 0)
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

#### Асимптоты:
{self._format_asymptotes()}
    
* Точки пересечения с осью x: ${latex(x_intersections)}$
    
* Точка пересечения с осью y: ${latex(y_intersection)}$

#### Производная функции:

* $f'(x) = {latex(self.f_prime)}$

* $ D(f'(x)) : {latex(continuous_domain(self.f_prime, self.x, S.Reals))}$
    
* $ E(f'(x)) : {safe_range(self.f_prime)}$
    
* Точки пересечения с осью x: ${latex(f_prime_intersections)}$
    
* Точка пересечения с осью y: ${latex(f_prime_y_intersection)}$

#### Экстремумы функции:
*    {self._analyze_extrema()}

#### Монотонность функции:
*    {self._analyze_monotonicity()}

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
*    {self._analyze_concavity()}

#### Особые точки функции:
*    {self._format_special_points()}
"""
        display(Markdown(description))

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
