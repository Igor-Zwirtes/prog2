import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Questão 1
class PolinomialAproximation:
    def __init__(self, degree):
        self.d = degree # Grau do polinômio
        self.coeficients = None # Coeficientes do polinômio

    def polinomial_aproximation(self, points):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        self.coeficients = np.polyfit(x, y, self.d)

    def eval_poly(self, x):
        result = 0
        for i, coef in enumerate(reversed(self.coeficients)): # Os coeficientes são salvos do maior grau para o menor
            result += coef * (x ** i)
        return result
    
    def __call__(self, x):
        return self.eval_poly(x)


# Questão 2
'''
A estratégia consiste em testar o polinômio gerado para cada valor possível de d até que a diferênça causada no R^2 com o aumento de d seja inferior a 0.1%, pois de acordo com testes empíricos feitos por mim mesmo, uma tolerância menor do que essa faz com que d seja um número muito baixo, normalmente 1 ou 2, enquanto tolerância maiores fazem com que o polinômio se torne muito instável, variando a ordens de grandeza como 1e7, embora os pontos tivessem distribuição uniforme em (-15,15). O motivo dessa estratégia é que, embora aumentando o grau do polinômio se obtenha uma aproximação melhor, para a maioria dos casos, caso o R^2 não aumente significativamente, então o aumento de complexidade não compensa o ganho de precisão, como por exemplo, alguns polinômios de grau muito alto, como 10, por exemplo, podem ser muito bem aproximados por um polinômio de grau 4, com um R^2 de 0.99, poupando recursos computacionais, além de que o enunciado da questão não especifica qual o critério para definir o melhor valor de d. Além disso, isso garante que o grau não aumente mais que o necessário, garantindo que um polinômio de grau 2, por exemplo, não tenha um d de 6.
'''

def find_degree(points):
    D = 1 # Melhor valor de d
    d = 1 # Contador para o grau que está sendo usado atualmente
    eps_max = 0 # Maior valor de R^2
    eps = 0 # Valor atual do R^2 para cada valor de d
    
    while True: # Aumenta o grau enquanto a variação de eps for maior que 0.001
        results = []
        poly = PolinomialAproximation(d) 
        poly.polinomial_aproximation(points) # Faz a aproximação usando o grau d
        
        for point in points:
            x, y = point
            results.append((x, y, poly(x))) # Salva os valores do ponto e da aproximação
        
        y_mean = sum(y for _, y in points) / len(points)
        RST = sum((y - y_mean) ** 2 for _, y in points)
        RSS = sum((y_actual - y_pred) ** 2 for _, y_actual, y_pred in results)
        
        eps = 1 - RSS / RST # Calcula o R^2

        if eps-eps_max<0.001 and eps_max != 0: # Verifica se o aumento do R^2 foi significativo
            return (D, eps)
        
        if eps > eps_max:
            eps_max = eps
            D = d
        d += 1

# Função para gerar polinômios para os testes com um pequeno erro nos valores
def data_gen(n, d):
    data = []
    coef = [random.uniform(-5, 5) for _ in range(d+1)]
    x = np.linspace(-15, 15, n)
    
    for xi in x:
        eps = random.normalvariate(0, 10)
        y = eps
        for k in range(d+1):
            y += coef[k] * (xi**k)
        data.append((xi, y))
    
    return data, coef[::-1] # Retorna os coeficientes no mesmo modelo da questão 1

# Função que faz o conjunto de testes solicitados
def testes(n=20):
    for i in range(1, n+1): # Valores de d que serão testados
        points, coef = data_gen(75, i) # Gera pontos e o os coeficientes do polinômio original
        d, R = find_degree(points)
        teste = PolinomialAproximation(d)
        teste.polinomial_aproximation(points)
        print(f'dimensão = {i}')
        print(f'R² = {R} | d encontrado = {d}')
        print(f'coeficientes corretos = {coef}')
        print(f'coeficientes encontrados = {teste.coeficients}')
        print('')

'''
Os testes mostram que embora o d costume ser um valor menor do que i para i maior que 6, o valor de R^2 quase sempre é superior a 0.8, e sendo geralmente superior a 0.99, o que indica que embora o grau encontrado seja menor do que o do polinômio original, a aproximação é suficientemente boa, exceto para retas, o que se deve pela proximidade entre o RSS e o RST, já que ambos serão retas.
'''
testes()

# Questão 3
# Função para usar o minimize
def function(params, x, y):
    a, b = params
    return np.sum(np.abs(a * x + b - y))

# (a)
def abs_aproximation(x, y): # Recebe as coordenadas de maneira separada
    result = minimize(function, x0=[0,0], args=(x, y))

    a, b = result.x  # result.x retorna os valores de a e b

    return a, b # Retorna os coeficientes

# (b)
# Código do GitHub
def generate_points(m):
    np.random.seed(1)
    a = 6
    b = -3
    x = np.linspace(0, 10, m)
    y = a*x + b + np.random.standard_cauchy(size=m)

    return (x,y)

# Roda o código nos pontos solicitados
def test():
    results = []
    for i in [64, 128, 256, 512, 1024]:
        points = generate_points(i)
        a, b = abs_aproximation(np.array(points[0]), np.array(points[1]))
        results.append((a,b))
    return results

print(test())

# (c)
# Utilizei o mesmo código visto em aula
def square_aproximation(x, y):
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxx = np.sum(x*x) 
    Sxy = np.sum(x*y)

    M = np.array([[Sx, len(x)], [Sxx, Sx]])
    T = np.array([Sy, Sxy])
    a, b = np.linalg.solve(M, T)

    return a, b

# (d)
def data_visualization(x, y, lin, sqr): 
    a1, b1 = lin # Coeficientes obtidos utilizando a função módulo
    a2, b2 = sqr # Coeficientes obtidos utilizando a função quadrado
    _, ax = plt.subplots()

    # Plota os pontos originais
    plt.scatter(x, y, color='red', label='Pontos Originais')

    # Plota a aproximação linear
    plt.plot([min(x), max(x)], [min(x)*a1 + b1, max(x)*a1 + b1], color='blue', label='Aproximação linear')

    # Plota a aproximação quadrática
    plt.plot([min(x), max(x)], [min(x)*a2 + b2, max(x)*a2 + b2], color='green', label='Aproximação quadrática')
    
    # Configurações adicionais do gráfico
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Aproximação Polinomial')
    plt.legend()
    plt.grid(True)

    # Gera os valores mostrados no cabeçalho para comparação
    RST = np.sum((y - np.mean(y))**2)

    RSS_lin = np.sum((y - (a1*x + b1))**2)
    R2_lin  = 1.0 - (RSS_lin/RST)

    RSS_sqr = np.sum((y - (a2*x + b2))**2)
    R2_sqr  = 1.0 - (RSS_sqr/RST)

    ax.set(title= f"Linear: RSS = {RSS_lin/len(X):0.1f} | R^2 = {R2_lin:0.2f} -- Quadrático RSS = {RSS_sqr/len(X):0.1f} | R^2 = {R2_sqr:0.2f}")
    
    plt.show()

'''
delta = 25.0
a, b = 2.0, 35.0
intv = [1, 100]
X = np.linspace(intv[0], intv[1], 60)
epslon = stats.norm(0, delta).rvs(X.shape)
Y = (a*X + b) + epslon
m0 = np.mean(Y)

lin = linear_aproximation(X, Y)
sqr = square_aproximation(X, Y)
data_visualization(X, Y, lin, sqr)
'''

# (e)
'''
Utilizando o valor absoluto, temos a vantagem que erros grandes no valor de y terão menos impacto no resultado final, já que usando o quadrado, esses erros são aumentados ao quadrado, enquanto utilizando a soma dos quadrados, a análise se torna mais fácil de ser feita, pois a função quadrática é diferenciável em todo ponto, ao contrário da função valor absoluto, logo, nesse caso, para resolver a minimização, basta derivar em a e em b, igualar ambas derivadas a 0 e reolver o sistema, quanto usando o valor absoluto, e mais complicado minimizar. Portanto, o quadrado é mais simples de se calcular no caso geral, enquanto o absoluto é mais consistente quando há algum ponto com um erro grande.
'''