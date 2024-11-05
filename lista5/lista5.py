# Questão 1
'''
A função retorna o elemento da lista data que tem a menor distância euclidiana até o ponto de entrada.
Sendo n a quantidade de pontos em data, então para fazer data - point serão necessárias 2n operações, já que cada ponto tem exatamente duas coordenadas e será feita uma subtração para cada uma das componentes dos n vetores, o que será O(n). Para calcular np.linalg.norm(Dt, axis=1), será necessário calcular n normas, o que também estará na ordem de O(n), já que serão 4 operações por vetor. Para obter np.argmin(d), será necessário percorrer as n normas e compará-las com o atual menor valor, então também será O(n). Portanto, find_nb é O(n)
'''

# Questão 2
import random

def generate_maze(m, n, room=0, wall=1, cheese ='.'):
    maze = [[wall] * (2*n+1) for _ in range(2*m+1)] # Cria o labirinto
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Cria as direções

    def iterative(x, y):
        stack = [] # Pilha com as possíveis direções para seguir e com a direção da posição anterior
        stack.append((x, y, 0, 0)) # Começa adicionando a posição inicial e nenhuma direção anterior

        while stack:
            actual = stack.pop() # Remove a última posição da pilha dos candidatos e a torna a posição atual
            random.shuffle(directions) # Escolhe uma ordem aleatória para as direções
            if maze[2*actual[0]+1][2*actual[1]+1] == wall: # Verifica se a posição atual já foi percorrida ou não
                maze[2*actual[0]+1][2*actual[1]+1] = room # Caso não tenha sido, define como sendo uma sala
                maze[2*actual[0]+1-actual[2]][2*actual[1]+1-actual[3]] = room # Cria um corredor entre a posição atual e a anterior

                for dx, dy in directions: # Verifica quais direções possuem caminhos válidos
                    nx = actual[0] + dx # Nova coordenada x de cada direção
                    ny = actual[1] + dy # Nova coordenada y de cada direção
                    if 0 <= nx < n and 0 <= ny < m: # Verifica se ao ir para cada direção, continuará dentro do labirinto
                        stack.append((nx, ny, dx, dy)) # Adiciona a direção como candidato a nova posição

    iterative(0,0) # Começa a criar o labirindo a partir da posição (1,1)
    while True: # Escolhe a posição do queijo
        i = int(random.uniform(0, 2*n))
        j = int(random.uniform(0, 2*m))
        if maze[i][j] == room:
            maze[i][j] = cheese 
            break

    return maze

def print_maze(maze):
    for row in maze:
        print(" ".join(map(str, row)))

m, n = 5, 5  # Grid size
random.seed(10110)
room = ' '
wall = 'H'
cheese = '*'
maze = generate_maze(m, n, room, wall, cheese)

# Questão 3
'''
Foi utilizada busca em profundidade, pois o código verifica todos os caminhos possíveis percorrendo até achar o queijo ou um local sem nenhuma direção para seguir, e então vai para o próximo caminho.
'''
def find_cheese(maze, cheese = '.'):
    if maze[1][1] == cheese:
        return cheese # Se o queijo estiver na posição inicial, o caminho consiste apenas no queijo
    wall = maze[0][0] # Define o que é a parede, pois (0,0) sempre é uma
    way = [[0] * len(maze[0]) for _ in range(len(maze))] # Matriz que determina quais posições já foram percorridas, para evitar que o caminho se torne cíclico

    def walk(x, y, path=''):
        way[x][y] = 1 # Salva que a posição atual já foi percorrida
        if maze[x][y] == cheese: # Quando a posição atual for o queijo, o caminho estará completo
            return path
        else:
            step = None
            # Para cada direção, verifica se a posição será uma parede e se a posição já foi percorrida
            if way[x-1][y] == 0 and maze[x-1][y] != wall:
                direction = walk(x-1, y, path+'N') # Repete o processo recursivamente, adicionando a direção que o passo foi dado
                if direction is not None: # Se não houver caminho até o queijo, desconsidera a direção
                    step = direction
            if way[x+1][y] == 0 and maze[x+1][y] != wall:
                direction = walk(x+1, y, path+'S')
                if direction is not None:
                    step = direction
            if way[x][y+1] == 0 and maze[x][y+1] != wall:
                direction = walk(x, y+1, path+'E')
                if direction is not None:
                    step = direction
            if way[x][y-1] == 0 and maze[x][y-1] != wall:
                direction = walk(x, y-1, path+'W')
                if direction is not None:
                    step = direction
            return step
    
    path = walk(1,1) # Iniciada o caminho em (1,1)
    way = [[' '] * len(maze[0]) for _ in range(len(maze))] # Matriz que exibirá o caminho
    x,y = 1,1

    for step in path: # Mostra a direção de cada passo na matriz
        if step == 'S':
            way[x][y] = 'v'
            x += 1
        if step == 'N':
            way[x][y] = '^'
            x -= 1
        if step == 'E':
            way[x][y] = '>'
            y += 1
        if step == 'W':
            way[x][y] = '<'
            y -= 1
    way[x][y] = cheese # Adiciona o queijo na posição final do caminho

    print_maze(maze)
    print_maze(way)

# find_cheese(maze, '*')

# Questão 4
class Graph:
    def __init__(G):
        G._vertices = {} # Dicionário que salva as arestas, onde a chave x representa o vértice x e cada elemento na lista associada a esse vértice representam os vértices para os quais x aponta
        G._values = {} # Dicionário que salva o valor de cada vértice

    def adjacent(G, x, y):
        return y in G._vertices[x] # Verifica se x tem uma aresta que aponta para y

    def neighbors(G, x):
        return G._vertices[x] # Retorna todos os vértices para os quais x aponta

    def add_vertex(G, x):
        if x in G._vertices.keys(): # Verifica se o vértice já existe
            return False
        else:
            G._vertices[x] = [] # Cria um vértice de chave x, inicialmente sem arestas
            G._values[x] = 0
            return True

    def remove_vertex(G, x):
        if x not in G._vertices.keys(): # Verifica se o vértice existe
            return False
        else:
            del G._vertices[x] # Remove a chave do vértice do dicionário
            for y in G._vertices.keys():
                G.remove_edge(y, x) # Remove o vértice da lista de arestas de todos outros vértices
            return True

    def add_edge(G, x, y):
        if G.adjacent(x, y): # Verifica se a aresta já existe
            return False
        else:
            G._vertices[x].append(y) # Adiciona a aresta à lista de arestas de x
            return True

    def remove_edge(G, x, y):
        if not G.adjacent(x, y): # Verifica se a aresta existe
            return False
        else:
            G._vertices[x] = [edge for edge in G._vertices[x] if edge != y] # Cria uma cópia da lista de arestas, mas sem y
            return True

    def get_vertex_value(G, x):
        return G._values[x]

    def set_vertex_value(G, x, v):
        G._values[x] = v


if __name__ == '__main__': # Teste unitários solicitados
    teste = Graph()
    print(teste.add_vertex(1))
    print(teste.add_vertex(2))
    print(teste.add_vertex(3))
    print(teste.add_vertex(10))
    print(teste.add_vertex(3))
    print(teste.add_edge(1, 2))
    print(teste.add_edge(1, 3))
    print(teste.add_edge(3, 2))
    print(teste.add_edge(1, 3))
    print(teste.adjacent(1, 3))
    print(teste.adjacent(2, 4))
    print(teste.neighbors(3))
    print(teste.remove_vertex(3))
    print(teste.remove_edge(1, 2))
    teste.set_vertex_value(1, 'Valor')
    teste.set_vertex_value(4, -4.5)
    print(teste.get_vertex_value(1))
    print(teste.get_vertex_value(4))