from mpi4py import MPI
from numpy import empty, array, int32, float64, zeros, arange, dot, random
from matplotlib.pyplot import style, figure, axes, show
import time

comm = MPI.COMM_WORLD
numprocs = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    start_time_1 = time.time()


def conjugate_gradient_method(A_part, b_part, x, N): # не распараллеливаем скалярные произведения векторов
    p = empty(N, dtype=float64)
    r = empty(N, dtype=float64)
    q = empty(N, dtype=float64)

    s = 1

    p = 0.

    while s <= N:

        if s == 1:
            r_temp = dot(A_part.T, dot(A_part, x) - b_part)
            comm.Allreduce([r_temp, N, MPI.DOUBLE],
                           [r, N, MPI.DOUBLE], op=MPI.SUM)
        else:
            r = r - q / dot(p, q)

        p = p + r / dot(r, r)

        q_temp = dot(A_part.T, dot(A_part, p))
        comm.Allreduce([q_temp, N, MPI.DOUBLE],
                       [q, N, MPI.DOUBLE], op=MPI.SUM)

        x = x - p / dot(p, q)

        s = s + 1

    return x


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
else:
    rcounts_M = None
    displs_M = None

M_part = array(0, dtype=int32)

comm.Scatter([rcounts_M, 1, MPI.INT], [M_part, 1, MPI.INT], root=0)

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

x_solve = zeros(N, dtype=float64)

if rank == 0:
    start_time = time.time()

x_solve = conjugate_gradient_method(A_part, b_part, x_solve, N)

if rank == 0:
    end_time = time.time()
    print('Execution time of the script: {:.4f} second \n'.format(end_time - start_time))

if rank == 0:
    epsilon = 1e-6
    count = 0
    for i in range(N):
        if abs(x[i] - x_solve[i]) >= epsilon:
            count += 1
    if count != 0:
        abspercent = 1 - ((N - count) / N)
    else:
        abspercent = 0
    # print('When N = {0}, epsilon = {1}, the number of incorrectly counted elements is {2} or {3}%'.format(N, epsilon, count, abspercent))

if rank == 0:
    end_time_1 = time.time()
    print('Full script execution time: {:.6f} seconds \n'.format(end_time_1 - start_time_1))
    # print('x = {0} \n x_solve = {1}'.format(x, x_solve))