import math
import matplotlib.pyplot as plt
import matplotlib

# 切换为图形界面显示的终端TkAgg
matplotlib.use('TkAgg')
# 单位
kg_dim = 1
M_J_dim = 1.898e27 * kg_dim
s_dim = 1
year_dim = 31536000 * s_dim
month_dim = year_dim / 12
day_dim = month_dim / 30
m_dim = 1
AU_dim = 1.496e11 * m_dim
pc_dim = 30835997962819660.8 * m_dim
M_dimension = 1 / kg_dim
T_dimension = 1 / s_dim
L_dimension = 1 / m_dim

kg = 1 * M_dimension
s = 1 * T_dimension
year = 31536000 * T_dimension
m = 1 * L_dimension
AU = 1.496e11 * L_dimension
pc = 30835997962819660.8 * L_dimension

c = 3e8 * L_dimension / T_dimension  # m/s


# TianQin
def TianQin_S_n(f):
    L = math.pow(3, 0.5) * 1e8 * L_dimension  # m
    f_xing = c / (2 * math.pi * L)
    S_n = 1 / L ** 2 * (
                1e-24 + 4 * 1e-30 / (2 * math.pi * f) ** 4 * (1 + 1e-4 / f)) * (1 + 6 / 10 * (f / f_xing) ** 2)
    # S_a = 1e-30 * L_dimension ** 2 / T_dimension ** 3  # m^2*s^-4*Hz^-1
    # S_x = 1e-24 * L_dimension ** 2 / T_dimension ** 3  # m^2*s^-4*Hz^-1
    # S_n = (4 * S_a * (1 + 1e-4 / f) / math.pow(2 * math.pi * f, 4) + S_x) / math.pow(L, 2)
    return math.sqrt(S_n)


# LISA
def LISA_S_n(f):
    P_OMS = 1.5e-11 ** 2 * (1 + (0.002 / T_dimension / f) ** 4) * L_dimension ** 2 * T_dimension
    P_acc = 3e-15 ** 2 * (1 + (0.0004 / T_dimension / f) ** 2) * (
                1 + (f / 0.008 / T_dimension) ** 4) * L_dimension ** 2 / T_dimension ** 3
    L = 2.5e9 * L_dimension
    # f_xing = c / (2 * math.pi * L)
    f_xing = 0.01909 / T_dimension
    a1,a2,a3,a4,f_k=[0.138,-221,521,1680,0.00113]  # 4year
    # a1, a2, a3, a4, f_k = [0.165,299,611,1340,0.00173]  # 2year
    # a1, a2, a3, a4, f_k = [0.171,292,1020,1680,0.00215]  # 1year
    # a1, a2, a3, a4, f_k = [0.133, 243, 482, 917, 0.00258]  # 0.5year

    S_c=9e-45*f**(-7/3)*math.exp(-f**a1+a2*f*math.sin(a3*f))*(1+math.tanh(a4*(f_k-f)))

    # S_n = 10 / 3 / L ** 2 * (P_OMS + 2 * (1 + (math.cos(f / f_xing)) ** 2) * P_acc / (2 * math.pi * f) ** 4) * (
    #             1 + 6 / 10 * (f / f_xing) ** 2)+S_c
    S_n = 1 / L ** 2 * (P_OMS + 2 * (1 + (math.cos(f / f_xing)) ** 2) * P_acc / (2 * math.pi * f) ** 4) * (
                1 + 6 / 10 * (f / f_xing) ** 2)
    return math.sqrt(S_n)

# Taiji
def Taiji_S_n(f):
    P_OMS = 8e-12 ** 2 * (1 + (0.002 / T_dimension / f) ** 4) * L_dimension ** 2 * T_dimension
    P_acc = 3e-15 ** 2 * (1 + (0.0004 / T_dimension / f) ** 2) * (
            1 + (f / 0.008 / T_dimension) ** 4) * L_dimension ** 2 / T_dimension ** 3
    L = 3e9 * L_dimension
    f_xing = c / (2 * math.pi * L)
    a1, a2, a3, a4, f_k = [0.138, -221, 521, 1680, 0.00113]  # 4year
    # a1, a2, a3, a4, f_k = [0.165,299,611,1340,0.00173]  # 2year
    # a1, a2, a3, a4, f_k = [0.171,292,1020,1680,0.00215]  # 1year
    # a1, a2, a3, a4, f_k = [0.133, 243, 482, 917, 0.00258]  # 0.5year

    S_c = 9e-45 * f ** (-7 / 3) * math.exp(-f**a1 + a2 * f * math.sin(a3 * f)) * (1 + math.tanh(a4 * (f_k - f)))
    # S_n = 10 / 3 / L ** 2 * (P_OMS + 2 * (1 + (math.cos(f / f_xing)) ** 2) * P_acc / (2 * math.pi * f) ** 4) * (
    #         1 + 6 / 10 * (f / f_xing) ** 2)+S_c
    S_n = 1 / L ** 2 * (P_OMS + 2 * (1 + (math.cos(f / f_xing)) ** 2) * P_acc / (2 * math.pi * f) ** 4) * (
            1 + 6 / 10 * (f / f_xing) ** 2)
    return math.sqrt(S_n)
    # S_n = (4 * S_a * (1 + 1e-4 / f) / math.pow(2 * math.pi * f, 4) + S_x) / math.pow(L, 2)


def DECIGO_S_n(f):
    f_xing = 7.36
    S_n = 7.05e-48 * (1 + (f/f_xing) **2) + ((4.8E-51) * f**(-4))/(1 + (f/f_xing) **2) + 5.33E-52 * f **(-4)
    S_n = S_n*3/4
    return math.sqrt(S_n)


def BBO_S_n(f):
    S_n = 2e-49 * f **2 + 4.58e-49 + 1.26e-51 * f **(-4)
    S_n = S_n*3/4
    return math.sqrt(S_n)



f_x = []
TianQin_S_n_y = []
LISA_S_n_y = []
Taiji_S_n_y = []
DECIGO_S_n_y = []
BBO_S_n_y = []

for i in range(5):
    x = math.pow(10, -2)
    f_x.append(x)
    TianQin_y = TianQin_S_n(x)
    TianQin_S_n_y.append(TianQin_y)
    Taiji_y = Taiji_S_n(x)
    Taiji_S_n_y.append(Taiji_y)
    LISA_y = LISA_S_n(x)
    LISA_S_n_y.append(LISA_y)
    DECIGO_y = DECIGO_S_n(x)
    DECIGO_S_n_y.append(DECIGO_y)
    BBO_y = BBO_S_n(x)
    BBO_S_n_y.append(BBO_y)
# for i in range(101):
#     x = 10 ** (-4 + 4 * i / 100)
#     f_x.append(x)
#     TianQin_y = TianQin_S_n(x)
#     TianQin_S_n_y.append(TianQin_y)
#     Taiji_y = Taiji_S_n(x)
#     Taiji_S_n_y.append(Taiji_y)
#     LISA_y = LISA_S_n(x)
#     LISA_S_n_y.append(LISA_y)
#     DECIGO_y = DECIGO_S_n(x)
#     DECIGO_S_n_y.append(DECIGO_y)
#     BBO_y = BBO_S_n(x)
#     BBO_S_n_y.append(BBO_y)
# print(f_x)
# print('TianQin', TianQin_S_n_y)
# print('Taiji', Taiji_S_n_y)
# print('LISA', LISA_S_n_y)
# print('DECIGO', DECIGO_S_n_y)
# print('BBO', BBO_S_n_y)

fig, ax = plt.subplots() # 创建图实例
ax.plot(f_x, TianQin_S_n_y, marker='o', label='TianQin')
ax.plot(f_x, Taiji_S_n_y, marker='o', label='Taiji')
ax.plot(f_x, LISA_S_n_y, marker='o', label='LISA')
ax.plot(f_x, DECIGO_S_n_y, marker='o', label='DECIGO')
ax.plot(f_x, BBO_S_n_y, marker='o', label='BBO')
plt.loglog()
ax.set_xlabel('f (Hz) ') #设置x轴名称
ax.set_ylabel('Spectral density' + '($\mathit{Hz}^{-1/2}$)') #设置y轴名称
ax.set_title('sensitivity curve') #设置图名
ax.legend() #自动检测要在图例中显示的元素，并且显示
plt.show(block=True)

