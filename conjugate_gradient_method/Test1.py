from mpi4py import MPI
from numpy import empty, array, int32, float64, zeros, arange, dot, linalg, random
from matplotlib.pyplot import style, figure, axes, show
from scipy.linalg import lstsq
import time
import sys


def conjugate_gradient_method(A, b, x, N):
    s = 1
    p = zeros(N)
    while s <= N:
        if s == 1:
            r = dot(A.T, dot(A, x) - b)
        else:
            r = r - q / dot(p, q)
        p = p + r / dot(r, r)
        q = dot(A.T, dot(A, p))
        x = x - p / dot(p, q)
        s = s + 1

    return x


'''f1 = open('in.dat', 'r')
N = int(f1.readline())
M = int(f1.readline())
f1.close()'''

start_time_1 = time.time()

# N = 6400
# M = 9600
N = 200
M = 300

# x = array(range(1, N + 1), dtype=float64)
x = empty(N, float64)
for i in range(N):
    x[i] = random.uniform(-1_000, 1_000)
# print('x = ', x)

A = empty((M, N), float64)
b = empty(M, float64)

# f2 = open('A.dat', 'r')
for j in range(M):
    for i in range(N):
        A[j, i] = random.random_sample(1)
        # A[j, i] = float(f2.readline())
# print('A = ', A)


'''with f2 as file:
    line_count = 0
    for line in file:
        line_count += 1'''
# f2.close()

# print("Количество строк в файле:", line_count)
# print('Размер массива А = ', sys.getsizeof(A) / 1024 ** 3, 'Гб')

'''f3 = open('b.dat', 'r')
for j in range(M):
    b[j] = float(f3.readline())'''

'''with f3 as file:
    line_count = 0
    for line in file:
        line_count += 1'''
# f3.close()

# print("Количество строк в файле:", line_count)

b = dot(A, x)
# print('b = ', b)

'''rr = empty(N, float64)
rr = dot(A.T, dot(A, x) - b)
print(rr)'''

x_solve = zeros(N)

start_time = time.time()

# x, residuals, rank, s = lstsq(A, b) # В начале значения скачут
# x = linalg.solve(A, b) # только для квадратной матрицы A
x_solve = conjugate_gradient_method(A, b, x_solve, N)

end_time = time.time()
print('Execution time of the script: {:.4f} seconds \n'.format(end_time - start_time))

# print('x = ', x, '\n', 'x_solve = ', x_solve)

epsilon = 1e-10
count = 0
for i in range(N):
    if abs(x[i] - x_solve[i]) >= epsilon:
        count += 1
if count != 0:
    abspercent = 1 - ((N - count) / N)
else:
    abspercent = 0
print('When N = {0}, epsilon = {1}, the number of incorrectly counted elements is {2} or {3}% \n'.format(N, epsilon, count, abspercent))

end_time_1 = time.time()
print('Full script execution time: {:.6f} seconds'.format(end_time_1 - start_time_1), '\n')

'''style.use('dark_background')
fig = figure()
ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
ax.set_xlabel('i')
ax.set_ylabel('x[i]')
ax.plot(arange(N), x, '-y', lw=3)
show()'''
