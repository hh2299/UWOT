from mpi4py import MPI
import numpy as np
from numpy import poly1d
import matplotlib.pyplot as plt
from pylab import mpl
import time

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
result = np.zeros(size - 1)


def draw(positions, f_y):
    plt.scatter(positions[0], positions[1], label="离散数据", color="red")
    plt.plot(positions[0], f_y(positions[0]))
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.title("拉格朗日插值拟合数据")
    plt.legend(loc="upper left")
    plt.savefig('parallel compute.png')


def cal_param(positions, k):
    position_num = positions.shape[1]
    pt = poly1d(1.0)
    for j in range(position_num):
        if k == j:
            continue
        fac = positions[0][k] - positions[0][j]
        pt *= poly1d([1.0, -positions[0][j]]) / fac
    pt = pt * positions[1][k]
    return pt.coef


if rank == 0:
    x = np.linspace(-1, 1, 20)
    y = 1 / (1 + 25 * x ** 2)
    p = np.stack([x, y])
    start = time.time()
    # 根进程：1.广播坐标信息
    # p = readPosition("position.txt")
    comm.bcast(p, root=0)
    print("rank %d: positions seeded" % (rank))
else:
    # 子进程
    p = comm.bcast(None, root=0)
    result = np.array(cal_param(p, rank - 1))
    # 补高位系数为0的
    if len(result) < size - 1:
        result = np.pad(result, (size - len(result) - 1, 0), 'constant', constant_values=(0, 0))


param_arr = comm.gather(result, root=0)

# 参数累加+画图
if rank == 0:
    param_arr = np.array(param_arr)
    xs_arr = param_arr.sum(axis=0)
    end = time.time()
    print('并行计算耗时：{}'.format(end - start))
    f = poly1d(xs_arr)
    draw(p, f)
