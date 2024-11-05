import numpy as np

# Questão 1
class VectorSpace:
    def __init__(self, dim: int, field):
        self.dim = dim
        self._field = field
        
    def getField(self):
        return self._field
    
    def getVectorSpace(self):
        return f'dim = {self.dim!r}, field = {self._field!r}'

    def __repr__(self):
        return self.getVectorSpace()
    
    def __mul__(self, f):
        raise NotImplementedError
    
    def __rmul__(self, f):
        return self.__mul__(f)
    
    def __add__(self, v):
        raise NotImplementedError


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


class Vector3D(RealVector):
    _dim = 3
    def __init__(self, coord):
        super().__init__(coord)

    @staticmethod
    def _builder(coord):
        return Vector3D(coord)
    
    def norm(self):
        return (self.coord[0]**2 + self.coord[1]**2 + self.coord[2]**2)**0.5

    def vector_product(self, other_vector):
        return Vector3D([self.coord[1]*other_vector.coord[2] - self.coord[2]*other_vector.coord[1], self.coord[2]*other_vector.coord[0] - self.coord[0]*other_vector.coord[2], self.coord[0]*other_vector.coord[1] - self.coord[1]*other_vector.coord[0]])
    
    def __eq__(self, other_vector):
        for i in range(len(self.coord)):
            if abs(self.coord[i] - other_vector.coord[i]) > 4*np.finfo(float).eps: # Compara a diferença com o épsilon da máquina
                return False
        return True
    
    def __ne__(self, other_vector):
        return not self.__eq__(other_vector)
    
    def __gt__(self, other_vector):
        diff = other_vector.norm() - self.norm()
        epsilon = 4 * np.finfo(float).eps
        if diff < -epsilon: # Compara a diferença com o épsilon da máquina
            return True
        else:
            return False
    
    def __ge__(self, other_vector):
        diff = other_vector.norm() - self.norm()
        epsilon = 4 * np.finfo(float).eps
        if diff < -epsilon: # Compara a diferença com o épsilon da máquina
            return True
        elif abs(diff) <= epsilon: # Verifica se as normas são iguais
            return True
        else:
            return False
    
    def __lt__(self, other_vector):
        diff = self.norm() - other_vector.norm()
        epsilon = 4 * np.finfo(float).eps
        if diff < -epsilon: # Compara a diferença com o épsilon da máquina
            return True
        else:
            return False
    
    def __le__(self, other_vector):
        diff = self.norm() - other_vector.norm()
        epsilon = 4 * np.finfo(float).eps
        if diff < -epsilon: # Compara a diferença com o épsilon da máquina
            return True
        elif abs(diff) <= epsilon: # Verifica se as normas são iguais
            return True
        else:
            return False
        
    
# Questão 2
'''
É necessário um bit para representar o sinal do número, além disso, a base possui p dígitos, onde cada um tem β valores possíveis, logo, para a mantissa são p*log2(β) bits, e o expoente pode variar ente emin e emax, então são (emax - emin + 1) valores possíveis, logo, o expoente precisa de log2(emax - emin + 1) bits.
Portanto, a quantidade de bits será 1 + p*log2(β) + log2(emax - emin + 1)
'''

# Questão 3
def eps():
    epsilon = 1.0
    while 1.0 + epsilon > 1.0:
        epsilon /= 2.0
    return epsilon * 2.0

# Questão 4
def big_eps():
    epsilon = 1.0
    while 1000000.0 + epsilon > 1000000.0:
        epsilon /= 2.0
    return epsilon * 2.0

print(eps())
print(big_eps())

'''
Essa diferença de 6 ordens de grandeza implica que para números grandes a precisão para operações envolvendo valores fracionários será menor, portanto será preferível utilizar números próximos de 1 para realizar operações que envolvam números com parte fracionária muito próximas de zero, além disso, é necessário considerar essa diferença entre os valores possíveis do épsilon quando são tratados erros de arredondamento, já que será diferentes de acordo com a ordem de grandeza do número.
'''

# Questão 5
'''
Eu criaria uma classe que utiliza como base o metro e salva a parte inteira (metros) do número em um atributo e a parte fracionária (milímetros) em outro, realizando a conversão dos milímetros excedentes em uma soma para metros, além de um método para representação de ambas as partes juntas, operações básicas e conversão de uma das partes para a outra. Assim seria possível realizar as operações mantendo precisão, já que o número decimal não estaria sujeito a erros de arredondamento, pois será tratado como um inteiro, além disso, para representar o número todo, basta juntar ambas as partes, e seria possível realizar operações entre números parte a parte.
'''
class FractionedNumber: # Apenas números naturais
    def __init__(self, int_part, float_part):
        self.int = int_part  # Metros
        self.float = float_part  # Milímetros (4 casas decimais)
        if self.float >= 10000: # Converte 1000 milímetros em 1 metro
            self.int += self.float // 10000
            self.float = self.float % 10000
        while float_part < 0:
            int_part -= 1 # Evita que a parte decimal seja negativa
            float_part += 10000

    def __add__(self, other):
        if isinstance(other, float): # Verifica se o outro número é float
            int_part = self.int + int(other)
            float_part = self.float + int((other - int(other))*10000)
            return FractionedNumber(int_part, float_part)
        int_part = self.int + other.int
        float_part = self.float + other.float
        return FractionedNumber(int_part, float_part)
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        if isinstance(other, float): # Verifica se o outro número é float
            int_part = self.int - int(other)
            float_part = self.float - int((other - int(other))*10000)
            return FractionedNumber(int_part, float_part)
        int_part = self.int - other.int
        float_part = self.float - other.float
        return FractionedNumber(int_part, float_part)
    
    def __rsub__(self, other):
        return self - other
    
    def __mul__(self, other):
        if isinstance(other, float): # Verifica se o outro número é float
            total_millimeters = (self.int * 10000 + self.float) * other
            int_part = int(total_millimeters // 10000)
            float_part = total_millimeters % 10000
            return FractionedNumber(int_part, float_part)

        self_total = self.int * 10000 + self.float
        other_total = other.int * 10000 + other.float
        total_millimeters = self_total * other_total // 10000
        int_part = int(total_millimeters // 10000)
        float_part = total_millimeters % 10000
        return FractionedNumber(int_part, float_part)
    
    def __rmul__(self, a: float):
        return self.__mul__(a)
    
    def __div__(self, other):
        if isinstance(other, float): # Verifica se o outro número é float
            total_millimeters = (self.int * 10000 + self.float) / other
            int_part = int(total_millimeters // 10000)
            float_part = total_millimeters % 10000
            return FractionedNumber(int_part, float_part)

        self_total = self.int * 10000 + self.float
        other_total = other.int * 10000 + other.float
        total_millimeters = self_total / other_total // 10000
        int_part = int(total_millimeters // 10000)
        float_part = total_millimeters % 10000
        return FractionedNumber(int_part, float_part)
    
    def __rdiv__(self, other):
        return self / other

    def __str__(self):
        return f'{self.int}.{int(self.float):04d}'