import numpy as np
import time

# Questão 1
class MyArray:
    def __init__(self):
        self.data = np.empty(2)
        self.capacity = 2 # Tamanho do array
        self.occuped = 0 # Quantidade de espaços sendo usados

    def append(self, item):
        if self.occuped == self.capacity:
            self.capacity *= 2 # Duplica a capacidade do array quando ele estiver cheio
            new_data = np.empty(self.capacity) 
            new_data[:self.occuped] = self.data[:self.occuped] # Copia os valores que já estavam no array
            self.data = new_data
        self.data[self.occuped] = item # Faz o append no primeiro espaço não usado
        self.occuped += 1

    def __len__(self):
        return self.occuped # Considera o tamanho do array como a quantidade de espaços usados, pois o restante é apenas memória reservada

    def __getitem__(self, index):
        return self.data[index]

    def __setitem__(self, index, value):
        self.data[index] = value

    def __iter__(self):
        return iter(self.data)
    
    def __contains__(self, item):
        return item in self.data

    def __str__(self):
        return str(self.data[:self.occuped]) # Não coloca na string os espaços vazios

# Comparação de tempo entre ambas as classes
myarray = MyArray()
pythonlist = []

start = time.time()
for i in range(100000):
    myarray.append(i)
end = time.time()
myarray_time = end - start

start = time.time()
for i in range(100000):
    pythonlist.append(i)
end = time.time()
python_time = end - start

print('Tempo do MyArray: ', myarray_time, 's')
print('Tempo da Lista Python: ', python_time, 's')


# Questão 2
class ToroArray(np.ndarray):
    def __new__(self, values):
        return np.array(values).view(self) # Cria um array

    def __getitem__(self, index):
        if isinstance(index, tuple): # Trata índices que tem dimensão maior que 1
            return super().__getitem__(tuple(i % len(self) for i in index))
        else:
            return super().__getitem__(index % len(self)) # O índice será ó modulo do índice passado com o tamanho do array

    def __setitem__(self, index, value):
        if isinstance(index, tuple): # Trata índices que tem dimensão maior que 1
            super().__setitem__(tuple(i % len(self) for i in index), value)
        else:
            super().__setitem__(index % len(self), value) # O índice será ó modulo do índice passado com o tamanho do array


# Questão 3
'''
Para mostrar que n^2 + 1000n = O(n^2), temos que mostrar que existe N natural e c real tal que n^2 + 1000n <= cn^2 para todo n >= N.
n^2 + 1000n <= cn^2 ->
n^2 - cn^2 + 1000n <= 0 ->
n^2(1-c) + 1000n <= 0 ->
n^2(c-1) - 1000n >= 0 ->
n(n(c-1) - 1000) >= 0 (Como n^2 se trata de uma função quadrática, então sempre existirá um intervalo (0,ε) onde n^2 + 1000n > n^2, portanto n != 0, logo, podemos dividir ambos lados da igualdade por n) ->
n(c-1) - 1000 >= 0 ->
n(c-1) >= 1000 ->
n >= 1000/(c-1).
Logo, N = 1000/(c-1), e como N precisa ser natural, c também deverá ser natural e maior que 1.
Portanto, para c = 2, temos que N = 1000, para c = 101, N = 10 e para c = 1001, N = 1.
'''

# Questão 4
'''
Suponha que g é O(f), então sabemos que existem N natural e c real tais que g(n) <= cf(n) para todo n >= N, além disso, tomando k = 1/c, temos que kg(n) <= kcf(n), então f(n) >= kg(n), e como c != 0, então k é um número real, logo, f é Ω(g).
Analogamente, suponha que f é Ω(g), portanto existem N natural e c real tais que f(n) >= cg(n) para todo n >= N, logo, tomando k = 1/c, temos que kf(n) >= kcg(n), então g(n) <= kf(n), e como c != 0, então k é um número real, logo, g é O(f).
Portanto, g é O(f) se e somente se f ́e Ω(g).
'''

# Questão 5
'''
Suponha que g é Θ(f), então g é O(f) e Ω(f), e com base na questão 4, sabemos que f será Ω(g), e também que f será O(g), logo, f é Θ(g).
Agora, suponha que f é Θ(g), então f é O(g) e Ω(g), e com base na questão 4, sabemos que g será Ω(f), e também que g será O(f), logo, g é Θ(f).
Portanto, g ́e Θ(f) se e somente se f ́e Θ(g).
'''