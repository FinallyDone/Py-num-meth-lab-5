import numpy as np
import math

STEP = 0.0002
EPS = 0.000001
NSEG = 1


def fn_k(x):
    return (4 - 0.1 * x) / (x ** 2 + N / 16)


def fn_q(x):
    return (x + 5) / (x ** 2 + 0.9 * N)


def fn_f(x):
    return (N + x) / 3.5


def phi0(x):
    return Nu1 + (Nu2 - Nu1) * np.sin(np.pi * (x - a) / 2 / (b - a))


def d_phi0(x):
    return (Nu2 - Nu1) * np.cos(np.pi * (x - a) / 2 / (b - a)) * (np.pi / 2 / (b - a))


def phi(k, x):
    return np.sin(k * np.pi * (x - a) / (b - a))


def d_phi(k, x):
    return np.cos(k * np.pi * (x - a) / (b - a)) * (k * np.pi / (b - a))


def inter_a(x, i, j):
    return fn_k(x) * d_phi(i, x) * d_phi(j, x) + fn_q(x) * phi(i, x) * phi(j, x)


def inter_b(x, i, j):
    return fn_f(x) * phi(i, x) - fn_k(x) * d_phi0(x) * d_phi(i, x) - fn_q(x) * phi0(x) * phi(i, x)


def left_rectangle(func, a, b, nseg, **kwargs):
    i, j = kwargs['indices']
    h = (b - a) / nseg
    x = [a + abs(h) * i for i in range(nseg)]

    return h * sum(map(lambda x: func(x, i, j), x[:-1]))


def get_a_ij(i, j):
    return left_rectangle(inter_a, a, b, NSEG, indices=(i, j))


def get_b_ij(i, j):
    return left_rectangle(inter_b, a, b, NSEG, indices=(i, j))


def solve_Ux(x, c):
    _sum = sum([c[i - 1, 0] * phi(i, x) for i in range(1, n)])
    return phi0(x) + _sum


# Метод Гауса с выбором главного элемента по столбцу
def method_gaus_col(matrix_A, matrix_F):
    # Определитель матрицы А
    det = 1
    # Длина столбца и строки
    len_col = len(matrix_A)
    len_row = len(matrix_A[0])
    # Берем длину строки
    for i in range(len_row):
        # Поиск максимального элемента в столбце
        max_elem = 0.0
        max_elem_index = 0
        # Берем длину столбца
        for j in range(len_col-i):
            if math.fabs(matrix_A[j+i][i]) > max_elem:
                max_elem = math.fabs(matrix_A[j+i][i])
                max_elem_index = j+i

        # Меняем 1 строку на строку с главным элементом в обоих матрицах
        if max_elem_index != 1:
            matrix_A[i], matrix_A[max_elem_index] = matrix_A[max_elem_index], matrix_A[i].copy()
            matrix_F[i], matrix_F[max_elem_index] = matrix_F[max_elem_index], matrix_F[i].copy()

        # Делим строку на главный элемент
        # Проверяем, не является ли 2 матрица вектором
        if len(matrix_F[0]) <= 1:
            matrix_F[i] /= matrix_A[i][i].copy()
        else:
            matrix_F[i:i+1, :] /= matrix_A[i][i].copy()
        matrix_A[i:i+1, :] /= matrix_A[i][i].copy()

        # Приводим к верхне-угольной матрице матрицу А
        if i != (len_col-1):
            if len(matrix_F[0]) <= 1:
                matrix_F[i+1:] -= matrix_F[i:i+1] * matrix_A[i+1:, i:i+1]
            else:
                matrix_F[i+1:, :] -= matrix_F[i:i+1, :] * matrix_A[i+1:, i:i+1]
            matrix_A[i+1:, :] -= matrix_A[i:i+1, :] * matrix_A[i+1:, i:i+1]

    # Вычитаем значения и получаем решения
    for i in range(len_col-1):
        for j in range(i+1):
            matrix_F[len_col - 2 - i] -= matrix_A[len_col-2-i][len_row-1-j] * matrix_F[len_col - 1 - j]
            matrix_A[len_col-2-i][len_row-1-j] = 0

    # Вектор решений
    return matrix_F


if __name__ == '__main__':
    N = 1
    n = 5
    a = 3 / 5 - N / 13
    b = 2 - N / 13
    Nu1 = 15 / (N + 3)
    Nu2 = -6 * N / 21
    matrix_A = np.zeros((n, n))
    vector_B = np.zeros((n, 1))

    # Находим шаг
    a_prev = left_rectangle(inter_b, a, (a + STEP), NSEG, indices=(1, 1))
    NSEG *= 2
    a_actual = left_rectangle(inter_b, a, (a + STEP), NSEG, indices=(1, 1))
    while True:
        a_prev = a_actual
        NSEG *= 2
        a_actual = left_rectangle(inter_b, a, (a + STEP), NSEG, indices=(1, 1))
        if(abs(a_prev - a_actual) < EPS):
            break

    # Находим матрицу A и вектор b
    for i in range(0, n):
        for j in range(0, n):
            matrix_A[i, j] = get_a_ij(i+1, j+1)
            vector_B[i] = get_b_ij(i+1, j+1)

    # Находим вектор С по методу Гаусса
    vector_C = method_gaus_col(matrix_A.copy(), vector_B.copy())

    print("\nМатрица A:")
    for el_a in matrix_A:
        print('', *map('{:8.6f}'.format, el_a))

    print("\nВектор b:")
    for el_b in vector_B:
        print('', *map('{:8.4f}'.format, el_b))

    print("\nВектор c:")
    for el_c in vector_C:
        print('', *map('{:8.4f}'.format, el_c))

    # Получаем Xi от а до b + шаг (т.е. включая b)
    vector_X = np.arange(a, b + (b - a) / 6, (b - a) / 6)
    print("\nВектор X:")
    print('', *map('{:8.4}'.format, vector_X))

    # Получим решения U(x)
    vector_Ux = [solve_Ux(x, vector_C) for x in vector_X]
    print("\nВектор решений U(x):")
    print('', *map('{:8.4}'.format, vector_Ux))

