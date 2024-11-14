# Questão 1
def find_judge(n, trust):
    if n == 1 and not trust:
        return 1

    # Criação de arrays para armazenar quantas pessoas confiam em cada uma e quantas cada uma confia
    trusted_by = [0] * (n + 1)
    trusts = [0] * (n + 1)

    # Preenchendo os arrays com as informações da matriz de confiança
    for a, b in trust:
        trusts[a] += 1
        trusted_by[b] += 1

    # Verificando se há algum candidato a juiz
    for i in range(1, n + 1):
        if trusted_by[i] == n - 1 and trusts[i] == 0:
            # É impossível que existam dois ou mais candidatos a juiz, pois caso houvessem dois por exemplo, um teria de confiar no outro para que ambos tivessem a confiança de todos, porém nesse caso, eles confiariam em alguém e não poderiam ser o juiz, logo, há no máximo 1 candidato a juiz
            return i

    return -1

# Questão 2
'''
b) Para mostrar que o problema da Minimum Spanning Tree é  Ω(nlog⁡(n)), podemos fazer uma redução de ordenação à MST, e com isso teremos um limite inferior não trivial.
Para isso, considere que temos n números reais que queremos ordenar, e criamos um grafo no plano 2D com n vértices, onde cada vértice tem coordenadas (xi,0) correspondendo ao i-ésimo número real xi​ da lista. Após isso, é contruído um grafo completo onde cada vértice é um dos pontos (xi​,0) e a aresta entre dois vértices (xi​,0) e (xj​,0) tem um peso igual a ∣xi​−xj​∣, que é a distância entre os pontos na reta dos números reais. Uma Minimum-Spanning Tree neste grafo conecta todos os vértices de forma que a soma dos pesos das arestas é minimizada. A MST desse grafo corresponde a uma forma de conectar os números que respeita a ordem de seus valores (já que a árvore geradora mínima com pesos baseados em ∣xi​−xj​∣ conecta os pontos em ordem crescente dos números para minimizar o peso total). A ordem dos vértices na MST pode ser usada para obter a ordenação dos números originais.
Portanto, a redução do problema de ordenação de números reais para o problema de calcular uma MST mostra que calcular a MST tem uma complexidade de tempo  de pelo menos Ω(nlog(n)).

c) O algoritmo implementado tem complexidade O(n²), pois a cada iteração, é necessário percorrer todos os vértices e atualizar todas as distâncias, portanto, o algoritmo não é ótimo, já que há algoritmos que conseguem resolver esse problema em O(nlog(n)), como o algoritmo de Kruskal, e portanto, o problema da MST é θ(nlog(n)), sendo essa uma cota ótima.
'''

# a)
def minimum_spanning_tree(points):
    n = len(points)
    in_mst = [False] * n  # Marca quais vértices já estão na MST
    min_edge = [float('inf')] * n  # Salva as menores distâncias conhecidas
    min_edge[0] = 0  # Começa no primeiro vértice
    parent = [-1] * n  # Para reconstruir a MST

    for _ in range(n):
        # Encontra o vértice com a menor aresta para adicionar à MST
        u = -1
        for i in range(n):
            if not in_mst[i] and (u == -1 or min_edge[i] < min_edge[u]):
                u = i

        in_mst[u] = True

        # Atualiza as menores distâncias conhecidas e os pais dos vértices adjacentes
        for v in range(n):
            if not in_mst[v]:
                dist = euclidean_distance(points[u], points[v])
                if dist < min_edge[v]:
                    min_edge[v] = dist
                    parent[v] = u

    # Retorna as arestas da MST como pares (u, v) utilizando os índices da lista points
    mst_edges = [(parent[i], i) for i in range(1, n) if parent[i] != -1]
    return mst_edges

def euclidean_distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

# Questão 3
'''
a) Analisando os valores, vemos que a temperatura e altitude estão relacionadas de forma linear. Fazendo interpolação utilizando os dois pontos com as temperaturas mais próximas de 0°C, obtemos, considerando x como a temperatura e a altitude sendo f(x)
f(x) = 800 + (x-3)/(-2-3) * (1000-800) = 800 + (3-x)/5 * 200 = 800 + 40(3-x) => f(x) = 920 - 40x
f(0) = 920
Portanto o avião provavelmente estava a 920 metros quando passou por 0°C

b) Da mesma forma que no item a), considerando os valores mais próximos de 700 metros, obtemos
f(x) = 600 + (x-5)/(3-5) * (800-600) = 600 + (5-x)/2 * 200 = 600 + 100(5-x) => f(x) = 1100 - 100x
f(x) = 700 => 700 = 1100 - 100x => 100x = 400 => x = 4
Portanto, à 700 metros, o avião provavelmente estava a 4°C
'''

# Questão 4
from typing import Any
import numpy as np
import matplotlib.pyplot as plt

epsilon = np.finfo(float).eps
    
class Domain:
    min = None
    max = None

    def __contains__(self, x):
        raise NotImplementedError
    
    def __repr__(self):
        raise NotImplementedError

    def __str__(self):
        return self.__repr__()
    
    def copy(self):
        raise NotImplementedError 


class Interval(Domain):
    def __init__(self, p1, p2):
        self.inff, self.supp = min(p1, p2), max(p1, p2)
    
    @property
    def min(self):
        return self.inff

    @property
    def max(self):
        return self.supp
    
    @property
    def size(self):
        return (self.max - self.min)
    
    @property
    def haf(self):
        return (self.max + self.min)/2.0
    
    def __contains__(self, x):
        return  np.all(np.logical_and(self.inff <= x, x <= self.supp))

    def __str__(self):
        return f'[{self.inff:2.4f}, {self.supp:2.4f}]' 

    def __repr__(self):
        return f'[{self.inf!r:2.4f}, {self.supp!r:2.4f}]'
    
    def copy(self):
        return Interval(self.inff, self.supp)


class RealFunction:
    f = None
    prime = None
    domain = None
    
    def eval_safe(self, x):
        if self.domain is None or x in self.domain:
            return self.f(x)
        else:
            raise Exception("The number is out of the domain")

    def prime_safe(self, x):
        if self.domain is None or x in self.domain:
            return self.prime(x)
        else:
            raise Exception("The number is out of the domain")
        
    def __call__(self, x) -> float:
        return self.eval_safe(x)
    
    def plot(self):
        fig, ax = plt.subplots()
        X = np.linspace(self.domain.min, self.domain.max, 100)
        Y = self(X)
        ax.plot(X,Y)
        return fig, ax


def bissect(f: RealFunction, 
            search_space: Interval, 
            erroTol: float = 1e-4, 
            maxItr: int = 1e4, 
            eps: float = 1e-6 ) -> Interval:
    count = 0
    ss = search_space.copy()
    err = ss.size/2.0
    fa, fb = f(ss.min), f(ss.max)
    if fa * fb > -eps:
        if abs(fa) < eps:
            return Interval(ss.min, ss.min)
        elif abs(fb) < eps:
            return Interval(ss.max, ss.max)
        else:
            raise Exception("The interval extremes share the same signal;\n employ the grid search method to locate a valid interval.")
    while count <= maxItr and err > erroTol:
        count += 1
        a, b, m =  ss.min, ss.max, ss.haf
        fa, fb, fm = f(a), f(b), f(m)
        if abs(fm) < eps:
            return Interval(m, m)
        elif fa * fm < -eps:
            ss = Interval(a, m)
        elif fb * fm < -eps:
            ss = Interval(m, b)
    return ss


def grid_search(f: RealFunction, domain: Interval = None, grid_freq = 8) -> Interval:
    if domain is not None:
        D = domain.copy()
    else:
        D = f.domain.copy()
    L1 = np.linspace(D.min, D.max, grid_freq)
    FL1 = f(L1)
    TI = FL1[:-1]*FL1[1:]
    VI = TI <= 0
    if not np.any(VI):
        return None
    idx = np.argmax(VI)
    return Interval(L1[idx], L1[idx+1])

# Função para o método de Newton-Raphson
def newton_root(f: RealFunction, x0: float = 0.0, iterations: int = 100, erroTol: float = 1e-6):
    x = x0
    for _ in range(iterations):
        fx = f(x)
        fpx = f.prime(x)
        
        # Verifica se a derivada é muito pequena
        if abs(fpx) < erroTol:
            raise ValueError("Derivative is too small, cannot proceed.")
        
        # Verifica se a função está suficientemente próxima de zero
        if abs(fx) <= erroTol:
            return x
        
        # Calcula o próximo valor de x usando a fórmula de Newton
        x = x - fx / fpx
    
    # Se o número máximo de iterações foi atingido
    raise ValueError("No root found within the specified iterations.")

# Exeplificação solicitada
if __name__ == '__main__':
    # Definindo um intervalo
    d = Interval(-1.0, 2.0)
    print(d)

    # Função de teste: f(x) = x^2 - 1 e f'(x) = 2x
    class funcTest(RealFunction):
        f = lambda self, x: np.power(x, 2) - 1
        prime = lambda self, x: 2 * x
        domain = Interval(-2, 2)

    # Criando uma instância da função
    ft = funcTest()

    # Usando grid_search para encontrar um intervalo válido
    ND = grid_search(ft, grid_freq=12)
    print(f"Grid Search Result: {ND}")

    # Usando bissect para encontrar a raiz dentro de um intervalo
    result_bissect = bissect(ft, search_space=ND)
    print(f"Bisection Method Result: {result_bissect}")

    # Usando newton_root para encontrar a raiz
    root_newton = newton_root(ft, x0=1.5)
    print(f"Newton's Method Root: {root_newton}")

# Questão 5
import numpy as np
from scipy.interpolate import lagrange
import time

class interpolater:
    def evaluate(self, X):
        raise NotImplementedError

    def __call__(self,  X):
        return self.evaluate(X)


class VandermondeMatrix(interpolater):
    def __init__(self, x, y):
        if len(x) != len(y):
            raise RuntimeError(f"Dimensions must be equal len(x) = {len(x)} != len(y) = {len(y)}")
        self.data = [x, y]
        self._degree = len(x) -1
        self._buildMatrix()
        self._poly = np.linalg.solve(self.matrix, self.data[1])

    def _buildMatrix(self):
        self.matrix = np.ones([self._degree+1, self._degree+1])
        for i, x in enumerate(self.data[0]):
            self.matrix[i, 1:] = np.multiply.accumulate(np.repeat(x, self._degree))
    
    def evaluate(self, X):
        r = 0.0
        for c in self._poly[::-1]:
            r = c+r*X
        return r


def random_sample(intv, N):
    r = np.random.uniform(intv[0], intv[1], N-2)
    r.sort()
    return np.array([intv[0]] + list(r) + [intv[1]])

def error_pol(f, P, intv, n = 1000):
    x = random_sample(intv, n)
    vectError = np.abs(f(x)-P(x))
    return np.sum(vectError)/n, np.max(vectError)

# Construtor do polinômio de Lagrange
class LagrangeInterpolator(interpolater):
    def __init__(self, x, y):
        if len(x) != len(y):
            raise RuntimeError(f"Dimensions must be equal len(x) = {len(x)} != len(y) = {len(y)}")
        self.data = [x, y]
        self.poly = lagrange(x, y) # Utiliza a função do Scipy para gerar o polinômio

    def evaluate(self, X):
        return self.poly(X)


def compare_construction_time(x_points, y_points):
    # Tempo de construção para VandermondeMatrix
    start_time = time.time()
    VandermondeMatrix(x_points, y_points)
    vandermonde_time = time.time() - start_time

    # Tempo de construção para LagrangeInterpolator
    start_time = time.time()
    LagrangeInterpolator(x_points, y_points)
    lagrange_time = time.time() - start_time

    print(f"Tempo de construção (VandermondeMatrix): {vandermonde_time:.10f} segundos")
    print(f"Tempo de construção (LagrangeInterpolator): {lagrange_time:.10f} segundos")