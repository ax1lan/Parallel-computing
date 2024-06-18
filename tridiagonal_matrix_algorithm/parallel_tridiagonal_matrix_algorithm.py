from mpi4py import MPI
from numpy import empty, array, zeros, int32, float64, random
import time

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    start_time_1 = time.time()

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


def parallel_tridiagonal_matrix_algorithm(a_part, b_part, c_part, d_part):
    N_part = len(d_part)

    for n in range(1, N_part):  # Ход Гауса
        coef = a_part[n] / b_part[n - 1]
        a_part[n] = -coef * a_part[n - 1]
        b_part[n] = b_part[n] - coef * c_part[n - 1]
        d_part[n] = d_part[n] - coef * d_part[n - 1]

    for n in range(N_part - 3, -1, -1):  # Обратный ход Гауса
        coef = c_part[n] / b_part[n + 1]
        c_part[n] = -coef * c_part[n + 1]
        a_part[n] = a_part[n] - coef * a_part[n + 1]
        d_part[n] = d_part[n] - coef * d_part[n + 1]

    if rank > 0:
        temp_array_send = array([a_part[0], b_part[0], c_part[0], d_part[0]], dtype=float64)
    if rank < numprocs - 1:
        temp_array_recv = empty(4, dtype=float64)

    # Отправляем первую строчку каждого из блоков в процессор с (rank - 1)
    # и соответственно каждый из процессоров ждет получение этой строки от процессора с (rank + 1).
    # Отметим, что 0-ой процессор не отправляет данные, а (numprocs - 1)-ый процессор не принимает данные
    if rank == 0:
        comm.Recv([temp_array_recv, 4, MPI.DOUBLE], source=1, tag=0, status=None)
    if rank in range(1, numprocs - 1):
        comm.Sendrecv(sendbuf=[temp_array_send, 4, MPI.DOUBLE], dest=rank - 1, sendtag=0,
                      recvbuf=[temp_array_recv, 4, MPI.DOUBLE], source=rank + 1, recvtag=MPI.ANY_TAG, status=None)
    if rank == numprocs - 1:
        comm.Send([temp_array_send, 4, MPI.DOUBLE], dest=numprocs - 2, tag=0)

    # На всех процессора кроме (numprocs - 1)-ого пересчитываем элементы (b_part, c_part, d_part) последней строки соответствующего блока
    if rank < numprocs - 1:
        coef = c_part[N_part - 1] / temp_array_recv[1]
        b_part[N_part - 1] = b_part[N_part - 1] - coef * temp_array_recv[0]
        c_part[N_part - 1] = - coef * temp_array_recv[2]
        d_part[N_part - 1] = d_part[N_part - 1] - coef * temp_array_recv[3]

    temp_array_send = array([a_part[N_part - 1], b_part[N_part - 1],
                             c_part[N_part - 1], d_part[N_part - 1]], dtype=float64)

    if rank == 0:
        A_extended = empty((numprocs, 4), dtype=float64)
    else:
        A_extended = None

    # На 0-ом процессоре собираем матрицу для последующего вычисления значений x_temp
    comm.Gather([temp_array_send, 4, MPI.DOUBLE], [A_extended, 4, MPI.DOUBLE], root=0)

    # Вызываем последовательный метод решения СЛАУ
    if rank == 0:
        x_temp = consecutive_tridiagonal_matrix_algorithm(
            A_extended[:, 0], A_extended[:, 1], A_extended[:, 2], A_extended[:, 3])
    else:
        x_temp = None

    # Распределяем найденные x_temp по соответствующим процессорам
    if rank == 0:
        rcounts_temp = empty(numprocs, dtype=int32)
        displs_temp = empty(numprocs, dtype=int32)
        rcounts_temp[0] = 1
        displs_temp[0] = 0
        for k in range(1, numprocs):
            rcounts_temp[k] = 2
            displs_temp[k] = k - 1
    else:
        rcounts_temp = None
        displs_temp = None

    if rank == 0:
        x_part_last = empty(1, dtype=float64)
        comm.Scatterv([x_temp, rcounts_temp, displs_temp, MPI.DOUBLE],
                      [x_part_last, 1, MPI.DOUBLE], root=0)
    else:
        x_part_last = empty(2, dtype=float64)
        comm.Scatterv([x_temp, rcounts_temp, displs_temp, MPI.DOUBLE],
                      [x_part_last, 2, MPI.DOUBLE], root=0)

    x_part = empty(N_part, dtype=float64)

    # На каждом процессоре вычисляем соответствующий вектор x_part
    if rank == 0:
        for n in range(N_part - 1):
            x_part[n] = (d_part[n] - c_part[n] * x_part_last[0]) / b_part[n]
        x_part[N_part - 1] = x_part_last[0]
    else:
        for n in range(N_part - 1):
            x_part[n] = (d_part[n] - a_part[n] * x_part_last[0]
                         - c_part[n] * x_part_last[1]) / b_part[n]
        x_part[N_part - 1] = x_part_last[1]

    return x_part


# Функциия задает в качестве элементов диагоналей матрицы A произвольные числа
def diagonals_preparation(N_part):
    a = empty(N_part, dtype=float64)
    b = empty(N_part, dtype=float64)
    c = empty(N_part, dtype=float64)
    for n in range(N_part):
        b[n] = random.random_sample(1)
        a[n] = random.random_sample(1)
        c[n] = random.random_sample(1)
    return a, b, c


# Определяем N - число компонент модельного вектора x
N = 1_000_000

ave, res = divmod(N, numprocs)
rcounts = empty(numprocs, dtype=int32)
displs = empty(numprocs, dtype=int32)
for k in range(0, numprocs):
    if k < res:
        rcounts[k] = ave + 1
    else:
        rcounts[k] = ave
    if k == 0:
        displs[k] = 0
    else:
        displs[k] = displs[k - 1] + rcounts[k - 1]

N_part = array(0, dtype=int32)
displ = array(0, dtype=int32)

comm.Scatter([rcounts, 1, MPI.INT], [N_part, 1, MPI.INT], root=0)
comm.Scatter([displs, 1, MPI.INT], [displ, 1, MPI.INT], root=0)

# Формируем на каждом mpi-процессе свои кусочки диагоналей
codiagonal_down_part, diagonal_part, codiagonal_up_part = diagonals_preparation(N_part)

# Задаём модельный вектор x, компонентами которого является 
# последовательность натуральных чисел от 1 до N (включительно)
if rank == 0:
    x = empty(N, float64)
    for i in range(N):
        x[i] = random.uniform(-100, 100)
    # x = array(range(1, N + 1), dtype=float64)
else:
    x = empty(N, dtype=float64)

# Передаём модельный вектор x всем mpi-процессам
comm.Bcast([x, N, MPI.DOUBLE], root=0)

# Умножаем матрицу А на модельный вектор x
# В результате получаем модельную правую часть,
# распределённую по всем mpi-процессам по кусочкам    
b_part = zeros(N_part, dtype=float64)
for n in range(N_part):
    if rank == 0 and n == 0:
        b_part[n] = diagonal_part[n] * x[displ + n] + codiagonal_up_part[n] * x[displ + n + 1]
    elif rank == numprocs - 1 and n == N_part - 1:
        b_part[n] = codiagonal_down_part[n] * x[displ + n - 1] + diagonal_part[n] * x[displ + n]
    else:
        b_part[n] = codiagonal_down_part[n] * x[displ + n - 1] + diagonal_part[n] * x[displ + n] + codiagonal_up_part[
            n] * x[displ + n + 1]

if rank == 0:
    start_time = time.time()

# Для сформированной матрицы А и модельной правой части b
# запускаем реализованный нами алгоритм решения СЛАУ с трёхдиагональной матрицей
x_part = parallel_tridiagonal_matrix_algorithm(codiagonal_down_part, diagonal_part, codiagonal_up_part, b_part)
# print('for rank = {0} \n x_part = {1}'.format(rank, x_part))

if rank == 0:
    end_time = time.time()
    print('Execution time of the script: {:.6f} seconds'.format(end_time - start_time), '\n')

x_solve = zeros(N)
comm.Gatherv(x_part, [x_solve, N_part, displs, MPI.DOUBLE], root=0)

if rank == 0:
    epsilon = 1e-10
    count = 0
    for i in range(N):
        if abs(x[i] - x_solve[i]) >= epsilon:
            count += 1
    print('When epsilon = {0}, the number of incorrectly counted elements is {1}'.format(epsilon, count))

if rank == 0:
    print('x = {0} \n x_solve = {1}'.format(x, x_solve))
    end_time_1 = time.time()
    print('Full script execution time: {:.6f} seconds'.format(end_time_1 - start_time_1))


# Выводим результат и убеждаемся, что на каждом mpi-процессе 
# результат вычислений совпадает с кусочком модельного вектора
#print('For rank={} : x_part={}, N_part={}'.format(rank, x_part, N_part))
