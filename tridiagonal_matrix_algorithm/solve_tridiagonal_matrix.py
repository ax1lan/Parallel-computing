from numpy import empty, array, zeros, int32, float64, random, diag, set_printoptions
import time

def consecutive_tridiagonal_matrix_algorithm(a, b, c, d):
    N = len(d)

    x = empty(N, dtype=float64)

    for n in range(1, N):
        coef = a[n] / b[n - 1]
        b[n] = b[n] - coef * c[n - 1]
        d[n] = d[n] - coef * d[n - 1]

    for n in range(N - 2, -1, -1):
        coef = c[n] / b[n + 1]
        d[n] = d[n] - coef * d[n + 1]

    for n in range(N):
        x[n] = d[n] / b[n]

    return x


# Функциия задает в качестве элементов диагоналей матрицы A произвольные числа
def diagonals_preparation(N):
    a = empty(N, dtype=float64)
    b = empty(N, dtype=float64)
    c = empty(N, dtype=float64)
    for n in range(N):
        b[n] = random.random_sample(1)
        a[n] = random.random_sample(1)
        c[n] = random.random_sample(1)
    return a, b, c

start_time_1 = time.time()

# Определяем N - число компонент модельного вектора x
N = 1_000_000

x = empty(N, float64)
for i in range(N):
    x[i] = random.uniform(-10, 10)

# Задаём модельный вектор x, компонентами которого является
# последовательность натуральных чисел от 1 до N (включительно)
# x = array(range(1, N + 1), dtype=float64)

'''f1 = open('xData.dat', 'w')
for j in range(N):
    f1.write(str(x[j]) + '\n')
f1.close()'''

codiagonal_down_part, diagonal_part, codiagonal_up_part = diagonals_preparation(N)

'''A = diag(diagonal_part) + diag(codiagonal_down_part[1:], -1) + diag(codiagonal_up_part[:-1], 1)
f2 = open('AData.dat', 'w')
for j in range(N):
    f2.write(str(A[j]) + '\n')
f2.close()'''

# Вычисление вектора b
b_vector = zeros(N, dtype=float64)
for n in range(N):
    if n == 0:
        b_vector[n] = diagonal_part[n] * x[n] + codiagonal_up_part[n] * x[n + 1]
    elif n == N - 1:
        b_vector[n] = codiagonal_down_part[n] * x[n - 1] + diagonal_part[n] * x[n]
    else:
        b_vector[n] = codiagonal_down_part[n] * x[n - 1] + diagonal_part[n] * x[n] + codiagonal_up_part[n] * x[n + 1]

'''f2 = open('bData.dat', 'w')
for j in range(N):
    f2.write(str(b_vector[j]) + '\n')
f2.close()'''

start_time = time.time()

x_solve = consecutive_tridiagonal_matrix_algorithm(codiagonal_down_part, diagonal_part, codiagonal_up_part, b_vector)

'''f3 = open('x_solveData.dat', 'w')
for j in range(N):
    f3.write(str(x_solve[j]) + '\n')
f3.close()'''

end_time = time.time()
print('Execution time of the script: {:.6f} seconds'.format(end_time - start_time))

epsilon = 1e-10
count = 0
for i in range(N):
    if abs(x[i] - x_solve[i]) >= epsilon:
        count += 1

print('При ε = {0} количество неверно подсчитанных элементов = {1}'.format(epsilon, count))
end_time_1 = time.time()
print('Full script execution time: {:.6f} seconds'.format(end_time_1 - start_time_1))



# A = create_tridiagonal_matrix(N)
# set_printoptions(linewidth=200)
# print(A)

# print(x, "\n", x_solve)