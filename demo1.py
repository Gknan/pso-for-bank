import numpy as np

class A():
    def __init__(self):
        self.a = 2
        self.b = 2

    def show_a(self, a=None):
        self.a = a or self.a
        print(self.a)


class B():
    def __init__(self, A):
        self.A = A
        self.mat = np.zeros((self.A.a, self.A.b))

    def show(self):
        print(self.mat.shape)

if __name__ == '__main__':
    a = np.zeros((46,1), dtype=int)
    b = np.random.uniform(low=-1, high=1, size=(2, 3, 3))
    print(b)
    b = np.around(b, decimals=2)
    print(b)
