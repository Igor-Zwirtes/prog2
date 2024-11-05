import scipy.stats
import scipy.spatial
import numpy as np
import random
import time

# Questão 1
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

    
def is_balanced(root):
    def height(node): # Função auxiliar para verificar a altura das subárvores
        if node == None: # Nós vazios não são folhas, portanto tem altura 0
            return 0, True # Retorna a altura e se o nó é balanceado
        
        left_height, left_balanced = height(node.left) # Calcula a altura do nó filho à esquerda
        if not left_balanced:
            return 0, False # Retorna a altura e se o nó é balanceado
        
        right_height, right_balanced = height(node.right) # Calcula a altura do nó filho à direita
        if not right_balanced:
            return 0, False # Retorna a altura e se o nó é balanceado
        
        if abs(left_height - right_height) > 1: # Verifica se a diferença de altura entre ambos nós é maior que 1
            return 0, False # Retorna a altura e se o nó é balanceado
        
        return max(left_height, right_height) + 1, True # A altura de um nó é a altura de seu filho com a maior altura mais 1

    _, balanced = height(root) # Verifica se a raiz é balanceada
    return balanced

# Questão 2
class MyStack:
    def __init__(self):
        self._data = []  # Lista que armazena os elementos da pilha

    def pop(self):
        if self.is_empty():
            raise ValueError('Pilha vazia')
        index = random.randint(0,len(self._data)-1) # Escolhe um índice aleatório
        self._data[-1], self._data[index] = self._data[index], self._data[-1] # Troca de posição o último item da pilha com o item escolhido
        return self._data.pop() # Realiza um método pop do Python, que é O(1) para o último elemento da pilha

    def push(self, item):
        self._data.append(item) # Adiciona um elemento ao final da pilha

    def size(self):
        return len(self._data) # Retona o tamanho da pilha

    def top(self):
        if self.is_empty():
            raise ValueError('Pilha vazia')
        return self._data[-1] # Retorna o item no topo da pilha, sem remover ele

    def is_empty(self):
        return len(self._data) == 0 # Retorna se a pilha está vazia

    def __str__(self):
        return str(self._data) # Retorna uma string dos itens da pilha

# Teste para verificar que os métodos push e pop tem O(1)
teste = MyStack()
for i in range(1000):
    t1 = time.time()
    teste.push(i)
    t2 = time.time()
    print('Tempo push: ', t2-t1, 'Tamanho: ', teste.size())
    
for _ in range(200):
    t1 = time.time()
    teste.pop()
    t2 = time.time()
    print('Tempo pop: ', t2-t1, 'Tamanho: ', teste.size())

# Questão 3
'''
a) Cada nó guardaria uma variável de valor e uma lista de ponteiros de seus nós filhos.
b) Para adicionar um novo filho, bastaria adicionar o novo ponteiro no final da lista, assim seria possível ter e acessar de maneira eficiente cada um dos filhos.
c) Para a travessia em profundidade, a lista de nós filhos seriam percorridas da esquerda para a direita, começando na raiz e percorrendo até uma folha, então sempre que um nó estivesse completo seria retornado para o anterior e se prosseguiria para o próximo filho, completando todos da esquerda para a direita. Já para a travessia em largura, seria criada uma lista com o nó raiz e um ponteiro, esse ponteiro iria percorrer essa lista e adicionaria todos os nós filhos do nó atual, na sequência da lista do atributo, para essa lista que está sendo percorrida.
'''

# Questão 4
def random_numbers(N, type):
    if type == 'Uniforme':
        x = scipy.stats.uniform.rvs(loc=-1, scale=2, size=N)
        y = scipy.stats.uniform.rvs(loc=-1, scale=2, size=N)
    
    elif type == 'Normal':
        x = scipy.stats.norm.rvs(loc=0, scale=0.5, size=N)
        y = scipy.stats.norm.rvs(loc=0, scale=0.5, size=N)

    elif type == 'Student t':
        # Como a questão não diz nada a respeito dos graus de liberdade para essa distribuição,
        # Escolhi arbitrariamente como sendo 2
        x = scipy.stats.t.rvs(df=2, loc=0, scale=0.5, size=N)
        y = scipy.stats.t.rvs(df=2, loc=0, scale=0.5, size=N)
    
    else:
        raise ValueError('Tipo incorreto')
    
    return np.column_stack((x, y)) # Retorna os pontos em uma lista do numpy
    
#print(random_numbers(15, 'Uniforme'))

# Questão 5
def convex_hull(points):
    hull = scipy.spatial.ConvexHull(points) # Calcula os vértices do fecho convexo
    return points[hull.vertices] # Retorna os pontos que estão na borda do fecho convexo, não repete o primeiro ponto no final

#print(convex_hull(random_numbers(20, 'Normal')))