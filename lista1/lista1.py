# Questão 1
def escada(n):
    # A quantidade de formas de subir uma escada de n degraus é igual ao número n da sequência de Fibonacci
    # Além disso, o código considera que uma escada deve ter ao menos um degrau, e um número natural de degraus
    if n < 1 or n != int(n):
        raise ValueError
    lista = []
    for i in range(1,n+1):
        if i == 1 or i == 2:
            lista.append(i)
        else:
            # Cada número da lista é obtido somando os dois números imediatamente anteriores
            lista.append(lista[i-2]+lista[i-3])
    return lista[n-1]

# Questão 2
# Classes que estavam no GitHub
class Field:
    pass


class VectorSpace:
    """VectorSpace:
    Abstract Class of vector space used to model basic linear structures
    """
    
    def __init__(self, dim: int, field: 'Field'):
        """
        Initialize a VectorSpace instance.

        Args:
            dim (int): Dimension of the vector space.
            field (Field): The field over which the vector space is defined.
        """
        self.dim = dim
        self._field = field
        
    def getField(self):
        """
        Get the field associated with this vector space.

        Returns:
            Field: The field associated with the vector space.
        """
        return self._field
    
    def getVectorSpace(self):
        """
        Get a string representation of the vector space.

        Returns:
            str: A string representing the vector space.
        """
        return f'dim = {self.dim!r}, field = {self._field!r}'
        # return self.__repr__()

    def __repr__(self):
        """
        Get a string representation of the VectorSpace instance.

        Returns:
            str: A string representing the VectorSpace instance.
        """
        # return f'dim = {self.dim!r}, field = {self._field!r}'
        return self.getVectorSpace()
    
    def __mul__(self, f):
        """
        Multiplication operation on the vector space (not implemented).

        Args:
            f: The factor for multiplication.

        Raises:
            NotImplementedError: This method is meant to be overridden by subclasses.
        """
        raise NotImplementedError
    
    def __rmul__(self, f):
        """
        Right multiplication operation on the vector space (not implemented).

        Args:
            f: The factor for multiplication.

        Returns:
            The result of multiplication.

        Note:
            This method is defined in terms of __mul__.
        """
        return self.__mul__(f)
    
    def __add__(self, v):
        """
        Addition operation on the vector space (not implemented).

        Args:
            v: The vector to be added.

        Raises:
            NotImplementedError: This method is meant to be overridden by subclasses.
        """
        raise NotImplementedError


# As operações solicitadas na questão 2 foram implementadas na classe RealVector, para que sejam extendidas à classe Vector3D da questão 3
# Também modifiquei a classe RealVector que estava no GitHub
class RealVector(VectorSpace):
    _field = float
    def __init__(self, coord):
        dim = len(coord)
        super().__init__(dim, self._field)
        self.coord = coord

    @staticmethod
    def _builder(coord):
        raise NotImplementedError
    
    def add(self, other_vector):
        coord = []
        for i in range(self.dim):
            coord.append(self.coord[i]+other_vector.coord[i])
        return self._builder(coord)
    
    def scalar_mul(self, a):
        coord = []
        for i in self.coord:
            coord.append(i*a)
        return self._builder(coord)
    
    def inner_product(self, other_vector):
        product = 0
        for i in range(self.dim):
            product += self.coord[i]*other_vector.coord[i]
        return product

    def __add__(self, other_vector):
        return self.add(other_vector)
    
    def __sub__(self, other_vector):
        return self.add(other_vector.scalar_mul(-1))
    
    def __mul__(self, a):
        return self.scalar_mul(a)
    
    def __rmul__(self, a):
        return self.scalar_mul(a)

    def __str__(self):
        return str(self.coord)
    
    def __abs__(self):
        norm = 0
        for i in range(self.dim):
            norm += self.coord[i]**2
        return norm**0.5
    
    def __neg__(self):
        return self.scalar_mul(-1)


class Vector2D(RealVector):
    _dim = 2
    def __init__(self, coord):
        if len(coord) != 2:
            raise ValueError
        super().__init__(coord)

    @staticmethod
    def _builder(coord):
        return Vector2D(coord)
    
    # Retorna um vetor perpendicular
    def CW(self):
        return Vector2D([-self.coord[1], self.coord[0]])
    
    # Retorna um vetor perpendicular
    def CCW(self):
        return Vector2D([self.coord[1], -self.coord[0]])
    

# Questão 3
class Vector3D(RealVector):
    _dim = 3
    def __init__(self, coord):
        super().__init__(coord)

    @staticmethod
    def _builder(coord):
        return Vector3D(coord)

    # Retorna o produto vetorial
    def vector_product(self, other_vector):
        return Vector3D([self.coord[1]*other_vector.coord[2] - self.coord[2]*other_vector.coord[1], self.coord[2]*other_vector.coord[0] - self.coord[0]*other_vector.coord[2], self.coord[0]*other_vector.coord[1] - self.coord[1]*other_vector.coord[0]])


class Polynomial(VectorSpace):
    _field = 'Polynomial'
    def __init__(self, coord):
        # Considera a dimensão de um polinômio de grau n sendo n+1
        dim = 0
        # Organiza as coordenadas do vetor em ordem crescente em relação ao grau
        keys = sorted(coord.keys())
        ord_coord = {key: coord[key] for key in keys}
        for key, val in ord_coord.items():
            if val != 0:
                if key + 1 > dim:
                    dim = key + 1
        super().__init__(dim, self._field)
        self.coord = ord_coord

    def mul_scalar(self, a):
        # Multiplica o coeficiente de cada coordenada pelo escalar a
        coords = {}
        for i in self.coord.keys():
            coords[i] = self.coord[i] * a
        return Polynomial(coords)
    
    def add(self, other_vector):
        coords = {}
        for i in self.coord.keys():
            if i in coords:
                coords[i] += self.coord[i]
            else:
                coords[i] = self.coord[i]
        for i in other_vector.coord.keys():
            if i in coords:
                coords[i] += other_vector.coord[i]
            else:
                coords[i] = other_vector.coord[i]
        keys = sorted(coords.keys())
        ord_coord = {key: coords[key] for key in keys}
        return Polynomial(ord_coord)
    
    # Avalia o valor do polinômio no ponto x=a
    def evaluate_poly(self, a):
        result = 0
        for i in self.coord.keys():
            result += self.coord[i] * a**i
        return result
    
    def __add__(self, other_poly):
        return self.add(other_poly)
    
    def __sub__(self, other_poly):
        return self.add(other_poly.mul_scalar(-1))
    
    def __mul__(self, a):
        return self.mul_scalar(a)
    
    def __rmul__(self, a):
        return self.mul_scalar(a)
    
    def __neg__(self):
        return self.mul_scalar(-1)
    
    def __str__(self):
        result = ''
        # Escreve o polinômio começando pelo maior coeficiente
        for i in reversed(self.coord.keys()):
            if self.coord[i] == 0:
                continue
            if i == 0:
                if self.coord[i] != 1 and self.coord[i] != -1:
                    result = result + f'+{self.coord[i]}'
                    continue
            if i == 1:
                if self.coord[i] != 1 and self.coord[i] != -1:
                    result = result + f'+{self.coord[i]}x'
                    continue
            if self.coord[i] == 1:
                if i == 0:
                    result = result + f'+1'
                    continue
                if i == 1:
                    result = result + f'+x'
                    continue
                result = result + f'+x^{i}'
            elif self.coord[i] == -1:
                if i == 0:
                    result = result + f'-1'
                    continue
                if i == 1:
                    result = result + f'-x'
                    continue
                result = result + f'-x^{i}'
            elif self.coord[i] > 0:
                result = result + f'+{self.coord[i]}x^{i}'
            else:
                result = result + f'{self.coord[i]}x^{i}'
        if result[0] == '+':
            result = result[1:]
        return result

    
# Questão 5
import collections
import random

Card = collections.namedtuple('Card', ['rank', 'suit'])

class FrenchDeck:
    ranks = [str(n) for n in range(2, 11)] + list('JQKA')
    suits = 'spades diamonds clubs hearts'.split()

    def __init__(self):
        self._cards = [Card(rank, suit) for suit in self.suits
        for rank in self.ranks]

    def __len__(self):
        return len(self._cards)

    def __getitem__(self, position):
        return self._cards[position]
    
    # Adicionado o método __setitem__, que permite a atribuição de novos valores a elementos já existentes do objeto
    def __setitem__(self, position, value):
        self._cards[position] = value


myDeck = FrenchDeck()
print(myDeck[1])
random.shuffle(myDeck)


if __name__ == '__main__':
    # Teste vetor 3D
    v1 = Vector3D([5,7,9])
    v2 = Vector3D([1,2,5])
    print('v1 = ', v1)
    print('v2 = ', v2)
    print('Soma: ', v1+v2)
    print('Produto por escalar: ', v1*3)
    print('Produto vetorial: ', v1.vector_product(v2))
    print('Produto interno: ', v1.inner_product(v2))
    print('Norma de v1: ', abs(v1))

    # Teste polinômio
    d1 = {
    5: 3,
    2: 1,
    1: 7,
    0: 2
    }

    d2 = {
        6: 1,
        2: 3,
        0: 2
    }

    p1 = Polynomial(d1)
    p2 = Polynomial(d2)
    print('p1 = ', p1)
    print('p2 = ', p2)
    print('Soma: ', p1+p2)
    print('Produto por escalar: ', p1*3)
    print('Valor de p2 em x=4: ', p2.evaluate_poly(4))