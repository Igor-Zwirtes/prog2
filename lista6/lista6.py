class Graph:
    def __init__(G):
        G._vertices = {}
        G._values = {}

    def adjacent(G, x, y):
        return y in G._vertices[x]

    def neighbors(G, x):
        return G._vertices[x]

    def add_vertex(G, x):
        if x in G._vertices.keys():
            return False
        else:
            G._vertices[x] = []
            G._values[x] = 0
            return True

    def remove_vertex(G, x):
        if x not in G._vertices.keys():
            return False
        else:
            del G._vertices[x]
            for y in G._vertices.keys():
                G.remove_edge(y, x)
            return True

    def add_edge(G, x, y):
        if G.adjacent(x, y):
            return False
        else:
            G._vertices[x].append(y)
            G._vertices[y].append(x)
            return True

    def remove_edge(G, x, y):
        if not G.adjacent(x, y):
            return False
        else:
            G._vertices[x] = [edge for edge in G._vertices[x] if edge != y]
            G._vertices[y] = [edge for edge in G._vertices[y] if edge != x]
            return True

    def get_vertex_value(G, x):
        return G._values[x]

    def set_vertex_value(G, x, v):
        G._values[x] = v

G = Graph()
G.add_vertex(1)
G.add_vertex(2)
G.add_vertex(3)
G.add_vertex(4)
G.add_vertex(5)
G.add_vertex(6)
G.add_edge(1,3)
G.add_edge(1,4)
G.add_edge(2,4)
G.add_edge(3,2)
G.add_edge(3,4)
G.add_edge(3,5)
G.add_edge(4,5)
G.add_edge(5,6)
G.set_vertex_value(1, 2)
G.set_vertex_value(2, None)
G.set_vertex_value(3, 2.1)
G.set_vertex_value(4, 'maçã')
G.set_vertex_value(5, 'abelha')
G.set_vertex_value(6, -7)

# Questão 1
def bfs(graph: Graph, vertex):
    queue = [vertex] # Utiliza uma fila para que os nós sejam visitados na ordem em que foram encontrados, garantindo que a busca seja em largura
    path = [vertex] # Lista ordenada dos nós visitados
    while queue:
        for edge in graph.neighbors(queue[0]):
            if edge not in path: # Verifica quais vizinhos do nó atual ainda não foram visitados
                path.append(edge) # Adiciona os nós ainda não visitados ao final do caminho
                queue.append(edge) # Adiciona os nós ainda não visitados ao final da fila de nós a serem visitados
        queue = queue[1:] # Remove o nó atual da fila
    return path

# print(bfs(G, 1))

# Questão 2
def property_search(graph: Graph, vertex, value):
    # Considera que o primeiro vértice a ser visitado será passado à função, como na função original
    # vertex = graph._vertices[0] # Caso o vértice não deva ser passado pela função
    queue = [vertex]
    path = [vertex]
    while queue:
        for edge in graph.neighbors(queue[0]):
            if graph.get_vertex_value(edge) == value: # Verifica se o nó atual possui a propriedade desejado
                return edge # Caso tenha, retorna o nó
            if edge not in path:
                path.append(edge)
                queue.append(edge)
        queue = queue[1:]
    return None # Caso não tenha encontrado a propriedade especificada, retorna None

# print(property_search(G, 1, 'maçã'))

m = [
    ['1', '1', '0', '0', '0'],
    ['0', '1', '0', '0', '1'],
    ['1', '0', '0', '1', '1'],
    ['0', '0', '0', '0', '0'],
    ['1', '0', '1', '1', '0'],
]

# Questão 3
def islands(matrix):
    islands_n = 0 # Quantidade de ilhas na matriz
    visited = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))] # Matriz que salva quais pontos já foram considerados parte de uma ilha, para que não haja repetição
    x, y = 0, 0 # Contador para percoorer as coordenadas da matriz
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] # Direções para as quais caso haja uma célula de valor 1 adjancente será considerada parte da mesma ilha, ou seja, horizontal, vertical e diagonal
    y_max, x_max = len(matrix), len(matrix[0])

    while y < y_max: # Percorre até o final da matriz
        if matrix[y][x] == '1' and not visited[y][x]: # Verifica se a célula atual é terra e não foi verificada anteriormente
            visited[y][x] = 1 # Salva que a célula atual foi visitada
            path = [(y,x)] # Inicia uma busca BFS, verificando todas as células adjacentes de cada ponto de terra da ilha
            while path:
                for dy,dx in directions: # Verifica cada uma das direções
                    ny = path[0][0] + dy
                    nx = path[0][1] + dx
                    if 0 <= ny < y_max and 0 <= nx < x_max: # Verifica se as novas coordenadas estão dentro da matriz
                        if matrix[ny][nx] == '1' and not visited[ny][nx]: # Verifica se a nova célula é terra e não foi verificada ainda
                            visited[ny][nx] = 1 # Marca que a nova célula foi visitada
                            path.append((ny,nx)) # Adiciona a nova célula a lista dos pontos que fazem parte da ilha
                path = path[1:] # Remove a célula atual da fila
            islands_n += 1

        if x < x_max - 1:
            x += 1 # Percorre horizontalmente até o final da linha
        else: # Quando chega ao final de uma linha, vai para a próxima
            x = 0
            y += 1

    return islands_n

# print(islands(m))

# Questão 4
def centroid_islands(matrix):
    island_cells = [] # Lista de coordenadas das ilhas
    x, y = 0, 0
    directions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    y_max, x_max = len(matrix), len(matrix[0])

    while y < y_max:
        if matrix[y][x] == '1' and (y,x) not in island_cells:
            island = [(y,x)]
            path = [(y,x)]
            while path:
                for dy,dx in directions:
                    ny = path[0][0] + dy
                    nx = path[0][1] + dx
                    if 0 <= nx < x_max and 0 <= ny < y_max:
                        if matrix[ny][nx] == '1' and (ny,nx) not in island:
                            island.append((ny,nx))
                            path.append((ny,nx))
                path = path[1:]

            for i in island:
                island_cells.append(i) # Salva em uma lista todos os pontos que fazem parte de uma ilha
            island_cells.append(None) # Separa as ilhas para busca posterior

        if x < x_max - 1:
            x += 1
        else:
            x = 0
            y += 1

    islands = [] # Lista das ilhas
    island = []
    for i in island_cells: # Separa cada ilha em uma lista diferente
        if i != None:
            island.append(i)
        else:
            islands.append(island)
            island = []

    bigger_island = max(islands,key=len) # Verifica qual é a maior ilha
    smallest_island = min(islands, key=len) # Verifica qual é a menor ilha

    start = min(bigger_island, key=lambda x: x[0])[0]
    end = max(bigger_island, key=lambda x: x[0])[0]
    centroid_x = (end+start)//2 # Calcula a coordenada x do centróide, ou seja, a média entre o ponto mais a oeste e o ponto mais ao leste
    start = min(bigger_island, key=lambda x: x[1])[1]
    end = max(bigger_island, key=lambda x: x[1])[1]
    centroid_y = (end+start)//2 # Calcula a coordenada y do centróide, ou seja, a média entre o ponto mais ao norte e o ponto mais ao sul
    bigger_centroid = (centroid_x, centroid_y) # Salva as coordenadas do centróide

    start = min(smallest_island, key=lambda x: x[0])[0]
    end = max(smallest_island, key=lambda x: x[0])[0]
    centroid_x = (end+start)//2
    start = min(smallest_island, key=lambda x: x[1])[1]
    end = max(smallest_island, key=lambda x: x[1])[1]
    centroid_y = (end+start)//2
    smaller_centroid = (centroid_x, centroid_y)

    return bigger_centroid, smaller_centroid

# print(centroid_islands(m))

n = [
    ['0', '1', '0'],
    ['1', '0', '1'],
    ['0', '1', '0']
]

# Questão 5
def lake(matrix):
    x, y = 0, 0
    has_lake = False
    visited = [[False for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    y_max, x_max = len(matrix), len(matrix[0])
    directions = [(-1,0),(0,-1),(1,0),(0,1)] # Não será necessário verificar as diagonais, apenas horizontalmente e verticalmente
    while y < y_max:
        if matrix[y][x] == '0' and not visited[y][x]:
            is_lake = True # Inicialmente, todas porções de água fazem parte de um lago, a menos que haja um caminho horizontal e/ou vertical dela até uma das bordas da matriz
            queue = [(y,x)]
            visited[y][x] = 1
            while queue:
                for dy,dx in directions:
                    nx = queue[0][1] + dx
                    ny = queue[0][0] + dy
                    if 0 <= nx < x_max and 0 <= ny < y_max:
                        if matrix[ny][nx] == '0':
                            queue.append((ny,nx))
                            visited[ny][nx] = 1
                    else: # Nesse caso, há um caminho até a borda da matriz
                        is_lake = False
                        break
                if not is_lake:
                    break
                queue = queue[1:]
            if is_lake: # Se não houve nenhum caminho da célula de água até a borda da matriz, ela será um lago
                has_lake = True
        
        if x < x_max - 1:
            x += 1
        else:
            x = 0
            y += 1

    return has_lake

# print(lake(m))
# print(lake(n))