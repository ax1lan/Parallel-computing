from mpi4py import MPI
from numpy import empty, array, int32, float64, zeros, arange, dot, random
from matplotlib.pyplot import style, figure, axes, show
import time

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    start_time_1 = time.time()

def conjugate_gradient_method(A_part, b_part, x_part,
                              N, N_part, rcounts_N, displs_N):
    x = empty(N, dtype=float64)
    p = empty(N, dtype=float64)

    r_part = empty(N_part, dtype=float64)
    p_part = empty(N_part, dtype=float64)
    q_part = empty(N_part, dtype=float64)

    ScalP = array(0, dtype=float64)
    ScalP_temp = empty(1, dtype=float64)

    s = 1
    p_part = 0.

    while s <= N:

        if s == 1:
            comm.Allgatherv([x_part, N_part, MPI.DOUBLE], # Собираем вектор x из x_part на всех процессорах
                            [x, rcounts_N, displs_N, MPI.DOUBLE])
            r_temp = dot(A_part.T, dot(A_part, x) - b_part)
            comm.Reduce_scatter([r_temp, N, MPI.DOUBLE], # Суммируем r_temp получая вектор "r", который сразу же раскидывается по частям r_part на соответствующие процессоры
                                [r_part, N_part, MPI.DOUBLE], # по функционалу = Reduce + Scatterv
                                recvcounts=rcounts_N, op=MPI.SUM)
        else:
            ScalP_temp[0] = dot(p_part, q_part)
            comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE], # Суммируем результаты скалярного произведения ScalP_temp, получая ScalP, и записываем его на все процессоры
                           [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
            r_part = r_part - q_part / ScalP

        ScalP_temp[0] = dot(r_part, r_part)
        comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE], # Суммируем результаты скалярного произведения ScalP_temp, получая ScalP, и записываем его на все процессоры
                       [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
        p_part = p_part + r_part / ScalP

        comm.Allgatherv([p_part, N_part, MPI.DOUBLE], # Собираем на всех процессорах вектор p из p_part
                        [p, rcounts_N, displs_N, MPI.DOUBLE])
        q_temp = dot(A_part.T, dot(A_part, p))
        comm.Reduce_scatter([q_temp, N, MPI.DOUBLE], # Суммируем q_temp получая вектор "q", который сразу же раскидывается по частям q_part на соответствующие процессоры
                            [q_part, N_part, MPI.DOUBLE],
                            recvcounts=rcounts_N, op=MPI.SUM)

        ScalP_temp[0] = dot(p_part, q_part)
        comm.Allreduce([ScalP_temp, 1, MPI.DOUBLE], # Суммируем результаты скалярного произведения ScalP_temp, получая ScalP, и записываем его на все процессоры
                       [ScalP, 1, MPI.DOUBLE], op=MPI.SUM)
        x_part = x_part - p_part / ScalP # Вычисляем очередное преближение для вектора x

        s = s + 1

    return x_part


# N = 6400
# M = 9600
N = 2000
M = 3000

if rank == 0:
    x = empty(N, float64)
    for i in range(N):
        x[i] = random.uniform(-10, 10)
    # print('x = ', x)



def auxiliary_arrays_determination(M, numprocs):
    ave, res = divmod(M, numprocs - 1)
    rcounts = empty(numprocs, dtype=int32)
    displs = empty(numprocs, dtype=int32)
    rcounts[0] = 0
    displs[0] = 0
    for k in range(1, numprocs):
        if k < 1 + res:
            rcounts[k] = ave + 1
        else:
            rcounts[k] = ave
        displs[k] = displs[k - 1] + rcounts[k - 1]
    return rcounts, displs


if rank == 0:
    rcounts_M, displs_M = auxiliary_arrays_determination(M, numprocs)
    rcounts_N, displs_N = auxiliary_arrays_determination(N, numprocs)
else:
    rcounts_M = None
    displs_M = None
    rcounts_N = empty(numprocs, dtype=int32)
    displs_N = empty(numprocs, dtype=int32)

M_part = array(0, dtype=int32)
N_part = array(0, dtype=int32)

comm.Scatter([rcounts_M, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)

comm.Bcast([rcounts_N, numprocs, MPI.INT], root=0)
comm.Bcast([displs_N, numprocs, MPI.INT], root=0)

if rank == 0:
    A = empty((M, N), float64)
    for j in range(M):
        for i in range(N):
            A[j, i] = random.uniform(0, 10)

    # print('A = ', A)

if rank == 0:
    for k in range(1, numprocs):
        A_part = empty((rcounts_M[k], N), dtype=float64)
        for j in range(rcounts_M[k]):
            p = 0
            for i in range(N):
                p = j + sum(rcounts_M[:k])
                A_part[j, i] = A[p, i]
        comm.Send([A_part, rcounts_M[k] * N, MPI.DOUBLE], dest=k, tag=0)
    A_part = empty((M_part, N), dtype=float64)
else:
    A_part = empty((M_part, N), dtype=float64)
    comm.Recv([A_part, M_part * N, MPI.DOUBLE], source=0, tag=0, status=None)

if rank == 0:
    b = empty(M, dtype=float64)
    b = dot(A, x)
    # print('b = ', b)
else:
    b = None

b_part = empty(M_part, dtype=float64)

comm.Scatterv([b, rcounts_M, displs_M, MPI.DOUBLE],
              [b_part, M_part, MPI.DOUBLE], root=0)

# Задаем начальное приближение для x (в данном случаи заполняем его нулями)
if rank == 0:
    x_solve = zeros(N, dtype=float64)
else:
    x_solve = None

x_part = empty(rcounts_N[rank], dtype=float64)

comm.Scatterv([x_solve, rcounts_N, displs_N, MPI.DOUBLE],
              [x_part, rcounts_N[rank], MPI.DOUBLE], root=0)

if rank == 0:
    start_time = time.time()

x_part = conjugate_gradient_method(A_part, b_part, x_part, # Параллельное вычисление x_part для каждого из процессоров методом сопряженных градиентов
                                   N, rcounts_N[rank], rcounts_N, displs_N)

if rank == 0:
    end_time = time.time()
    print('Execution time of the script: {:.6f} seconds'.format(end_time - start_time))

comm.Gatherv([x_part, rcounts_N[rank], MPI.DOUBLE],
             [x_solve, rcounts_N, displs_N, MPI.DOUBLE], root=0)

if rank == 0:
    epsilon = 1e-14
    count = 0
    for i in range(N):
        if abs(x[i] - x_solve[i]) >= epsilon:
            count += 1
    if count != 0:
        abspercent = 1 - ((N - count) / N)
    else:
        abspercent = 0
    print('When N = {0}, epsilon = {1}, the number of incorrectly counted elements is {2} or {3}% \n'.format(N, epsilon, count, abspercent))

if rank == 0:
    print('x = {0} \n\n x_solve = {1} \n'.format(x, x_solve))
    end_time_1 = time.time()
    print('Full script execution time: {:.6f} seconds'.format(end_time_1 - start_time_1))

'''if rank == 0:
    style.use('dark_background')
    fig = figure()
    ax = axes(xlim=(0, N), ylim=(-1.5, 1.5))
    ax.set_xlabel('i');
    ax.set_ylabel('x[i]')
    ax.plot(arange(N), x, '-y', lw=3)
    # show()'''
