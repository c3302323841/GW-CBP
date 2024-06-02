import math
import numpy as np
from scipy import linalg
from scipy import integrate
import matplotlib
import matplotlib.pyplot as plt

# import pandas as pd
# 切换为图形界面显示的终端TkAgg
matplotlib.use('TkAgg')
# 单位——————————————————————————————————————————————————————————————————————————————————————————————————————————
kg_dim = 1
m_dim = 1
AU_dim = 1.496e11 * m_dim
pc_dim = 30835997962819660.8 * m_dim
s_dim = 1
year_dim = 31536000 * s_dim
month_dim = year_dim / 12
day_dim = month_dim / 30
# 单位选择———————————————————————————————————————————————————————————————————————————————————————————————————————
M_dimension = 1 / kg_dim
L_dimension = 1 / m_dim
T_dimension = 1 / s_dim
# 常量——————————————————————————————————————————————————————————————————————————————————————————————————————————
M_J = 1.898e27 * M_dimension  # 木星质量
M_sun = 2.0e30 * M_dimension  # 太阳质量
G = 6.67e-11 * L_dimension / m_dim ** 3 / (M_dimension * T_dimension ** 2)  # m^3/(kg * s^2)
c = 3e8 * L_dimension / m_dim / T_dimension  # m/s
d = 2062650000 * 1.496e11 * L_dimension  # 源光度距离
R_earth = 1.496e11 * L_dimension  # 地球半径
P_earth = 31536000 * T_dimension  # 地球公转周期
year = 31536000 * T_dimension
# 参数设定———————————————————————————————————————————————————————————————————————————————————————————————————————
f = 0.0075 / T_dimension  # 引力波频率（10 mHz）s-1
T_obs = 4 * 31536000 * T_dimension  # 观测周期
M_b = 1 * M_sun

p_na = 14  # x轴的P的数量
p_nb = 200  # 寻找最小Mp的上限
p_nc = 1
# nc: tianqin is 1, taiji ji 2, lisa is 3, tianqin + taiji is 4 tianqin + lisa is 5,
# lisa +taiji is 6, three is 7
# 常数——————————————————————————————————————————————————————————————————————————————————————————————————————————
psi_0 = 0  # rad 康亚城设定的常数,观测者初相位
phi_0 = 0  # 源轨道初相位

theta_s_h = math.acos(0.3)  # 源的坐标
phi_s_h = 5  # 源的坐标
theta_s = math.acos(0.3)  # 源的坐标
phi_s = 5  # 源的坐标

theta_l = math.acos(-0.2)  # 轨道坐标
phi_l = 4  # 轨道坐标
MM = math.pow(0.25 * M_b * M_b, 3 / 5) / math.pow(M_b, 0.2)  # 啁啾质量
f_dao = 96 / 5 * math.pow(G * MM / math.pow(c, 3), 5 / 3) * math.pow(math.pi, 8 / 3) * math.pow(f, 11 / 3)
# 功率谱密度—————————————————————————————————————————————————————————————————————————————————————————————————————
l_tian_qin = math.pow(3, 0.5) * 1e8 * L_dimension  # m
f_tian_qin = c / (2 * math.pi * l_tian_qin)
s_n_tianqin = 10 / 3 / l_tian_qin ** 2 * (
        (1e-24 * L_dimension ** 2 * T_dimension) + 4 * (1e-30 * L_dimension ** 2 / T_dimension ** 3) / (
        2 * math.pi * f / T_dimension) ** 4 * (1 + 1e-4 / f)) * (1 + 6 / 10 * (f / f_tian_qin) ** 2)
p_oms_lisa = 1.5e-11 ** 2 * (1 + (0.002 / T_dimension / f) ** 4) * L_dimension ** 2 * T_dimension
p_acc_lisa = 3e-15 ** 2 * (1 + (0.0004 / T_dimension / f) ** 2) * (
        1 + (f / 0.008 / T_dimension) ** 4) * L_dimension ** 2 / T_dimension ** 3
l_lisa = 2.5e9 * L_dimension
f_lisa = 0.01909 / T_dimension
s_n_lisa = 10 / 3 / l_lisa ** 2 * (
        p_oms_lisa + 2 * (1 + (math.cos(f / f_lisa)) ** 2) * p_acc_lisa / (2 * math.pi * f) ** 4) * (
                   1 + 6 / 10 * (f * f_lisa) ** 2)
p_oms_taiji = 8e-12 ** 2 * (1 + (0.002 / T_dimension / f) ** 4) * L_dimension ** 2 * T_dimension
p_acc_taiji = 3e-15 ** 2 * (1 + (0.0004 / T_dimension / f) ** 2) * (
        1 + (f / 0.008 / T_dimension) ** 4) * L_dimension ** 2 / T_dimension ** 3
l_taiji = 3e9 * L_dimension
f_taiji = c / (2 * math.pi * l_taiji)
s_n_taiji = 10 / 3 / l_taiji ** 2 * (
        p_oms_taiji + 2 * (1 + (math.cos(f / f_taiji)) ** 2) * p_acc_taiji / (2 * math.pi * f) ** 4) * (
                    1 + 6 / 10 * (f * f_taiji) ** 2)
f_decigo = 7.36
s_n_decigo = (3 / 4 * 7.05e-48 * (1 + (f / f_decigo) ** 2) + (4.8E-51 * f ** (-4)) / (
        1 + (f / f_decigo) ** 2) + 5.33E-52 * f ** (-4))
s_n_bbo = 2e-49 * f ** 2 + 4.58e-49 + 1.26e-51 * f ** (-4) * 3 / 4


# fisher矩阵构造————————————————————————————————————————————————————————————————————————————————————————————————
def fisher1(p_S_n_ni, p_P, p_M_p):
    # n1,h2构造————————————————————————————————————————————————————————————————————————————————————————————————
    def psi_s(p_theta_l, p_phi_l, p_theta_s, p_phi_s):
        return math.atan((math.cos(p_theta_s) * math.cos(p_phi_s - p_phi_l) * math.sin(p_theta_l) - math.cos(
            p_theta_l) * math.sin(p_theta_s)) / (math.sin(p_theta_l) * math.sin(p_phi_s - p_phi_l)))

    def i(p_theta_l, p_phi_l, p_theta_s, p_phi_s):
        return math.cos(p_theta_l) * math.cos(p_theta_s) + math.sin(p_theta_l) * math.sin(
            p_theta_s) * math.cos(p_phi_l - p_phi_s)

    theta_s_t = math.acos(0.08748875)
    def phi_s_t(t):
        return 0.466982 + 2e-5 * t

    K = math.pow(2 * math.pi * G / p_P, 1 / 3) * p_M_p / math.pow(p_M_p + M_b, 2 / 3) * math.sin(
        i(theta_l, phi_l, theta_s, phi_s))

    def F1_jia(p_theta_s, p_phi_s, p_psi_s):
        return 0.5 * (1 + math.cos(p_theta_s) * math.cos(p_theta_s)) * math.cos(2 * p_phi_s) * math.cos(
            2 * p_psi_s) - math.cos(
            p_theta_s) * math.sin(2 * p_phi_s) * math.sin(2 * p_psi_s)

    def F1_cha(p_theta_s, p_phi_s, p_psi_s):
        return 0.5 * (1 + math.cos(p_theta_s) * math.cos(p_theta_s)) * math.cos(2 * p_phi_s) * math.cos(
            2 * p_psi_s) + math.cos(
            p_theta_s) * math.sin(2 * p_phi_s) * math.cos(2 * p_psi_s)

    def F2_jia(p_theta_s, p_phi_s, p_psi_s):
        return 0.5 * (1 + math.cos(p_theta_s) * math.cos(p_theta_s)) * math.cos(
            2 * p_phi_s - 0.5 * math.pi) * math.cos(
            2 * p_psi_s) - math.cos(p_theta_s) * math.sin(2 * p_phi_s - 0.5 * math.pi) * math.sin(2 * p_psi_s)

    def F2_cha(p_theta_s, p_phi_s, p_psi_s):
        return 0.5 * (1 + math.cos(p_theta_s) * math.cos(p_theta_s)) * math.cos(
            2 * p_phi_s - 0.5 * math.pi) * math.cos(
            2 * p_psi_s) + math.cos(p_theta_s) * math.sin(2 * p_phi_s - 0.5 * math.pi) * math.cos(2 * p_psi_s)

    AA = 2 * math.pow(G * M_b
                      , 5 / 3) * math.pow(math.pi * f, 2 / 3) / (math.pow(c, 4) * d)

    def A_jia(p_i):
        return AA * (1 + math.pow(math.cos(p_i), 2))

    def A_cha(p_i):
        return 2 * AA * math.cos(p_i)

    def A1(p_i, p_theta_s, p_phi_s, p_psi_s):
        return math.pow(
            math.pow(A_jia(p_i), 2) * math.pow(F1_jia(p_theta_s, p_phi_s, p_psi_s), 2) + math.pow(A_cha(p_i),
                                                                                                  2) * math.pow(
                F1_cha(p_theta_s, p_phi_s, p_psi_s), 2), 1 / 2)

    def A2(p_i, p_theta_s, p_phi_s, p_psi_s):
        return math.pow(
            math.pow(A_jia(p_i), 2) * math.pow(F2_jia(p_theta_s, p_phi_s, p_psi_s), 2) + math.pow(A_cha(p_i),
                                                                                                  2) * math.pow(
                F2_cha(p_theta_s, p_phi_s, p_psi_s), 2), 1 / 2)

    def fi_1(p_i, p_theta_s, p_phi_s, p_psi_s):
        return math.atan(
            -1 * A_cha(p_i) * F1_cha(p_theta_s, p_phi_s, p_psi_s) / A_jia(p_i) / F1_jia(p_theta_s, p_phi_s,
                                                                                        p_psi_s))

    def fi_2(p_i, p_theta_s, p_phi_s, p_psi_s):
        return math.atan(
            -1 * A_cha(p_i) * F2_cha(p_theta_s, p_phi_s, p_psi_s) / A_jia(p_i) / F2_jia(p_theta_s, p_phi_s,
                                                                                        p_psi_s))

    def fi_t(t):
        return 2 * math.pi * t / P_earth + phi_0

    def f_obs(t):
        return (1 - K * math.cos(2 * math.pi * t / p_P + phi_0) / c) * (f + f_dao * t)

    def fiD(t):
        return 2 * math.pi * f_obs(t) * R_earth * math.sin(theta_s) * math.cos(
            phi_0 + 2 * math.pi * t / P_earth - phi_s) / c

    def psi_obs(t):
        return 2 * math.pi * (f + 0.5 * f_dao * t) * t - p_P * f * K / c * math.sin(
            fi_t(t)) - p_P * f_dao * t / c * K * math.sin(fi_t(t)) - p_P * p_P * f_dao * K / (
                2 * math.pi * c) * math.cos(
            fi_t(t))

    def h1(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.cos(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t))

    def h2(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.cos(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t))

    def find_delta_h1_theta_s():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s_t + delta, phi_s_t(x),
                                              psi_s(theta_l, phi_l, theta_s + delta, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s_t + delta, phi_s_t(x),
                                  psi_s(theta_l, phi_l, theta_s + delta, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s_t - delta, phi_s_t(x),
                                 psi_s(theta_l, phi_l, theta_s - delta, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s_t - delta, phi_s_t(x),
                                  psi_s(theta_l, phi_l, theta_s - delta, phi_s)) + fiD(
                    x))) / (
                    2 * delta)
        n1 = 0.01
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, n1) * par_1(x, n1)
            v1 = integrate.quad(y, 0, T_obs)[0]
            n2 = 0.1 * n1
            y1 = lambda x: par_1(x, n2) * par_1(x, n2)
            v2 = integrate.quad(y1, 0, T_obs)[0]
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            n1 = n2
            # print(n1)
        h = n1
        return h

    delta_h1_theta_s = find_delta_h1_theta_s()

    def find_delta_h1_phi_s():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s_t, phi_s_t(x) + delta,
                                              psi_s(theta_l, phi_l, theta_s, phi_s + delta)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s_t, phi_s_t(x) + delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s + delta)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s_t, phi_s_t(x) - delta,
                                 psi_s(theta_l, phi_l, theta_s, phi_s - delta)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s_t, phi_s_t(x) - delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s - delta)) + fiD(
                    x))) / (
                    2 * delta)

        n1 = 0.01
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, n1) * par_1(x, n1)
            v1 = integrate.quad(y, 0, T_obs)[0] * 2 * p_S_n_ni
            n2 = 0.1 * n1
            y1 = lambda x: par_1(x, n2) * par_1(x, n2)
            v2 = integrate.quad(y1, 0, T_obs)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            n1 = n2
            # print(n1)
        h = n1
        return h

    delta_h1_phi_s = find_delta_h1_phi_s()

    def find_delta_h1_theta_l():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                              psi_s(theta_l + delta, phi_l, theta_s, phi_s,
                                                    )) * math.cos(
                psi_obs(x) + fi_1(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l + delta, phi_l, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                 psi_s(theta_l - delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l - delta, phi_l, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        n1 = 0.01
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, n1) * par_1(x, n1)
            v1 = integrate.quad(y, 0, T_obs
                                )[0]
            n2 = 0.1 * n1
            y1 = lambda x: par_1(x, n2) * par_1(x, n2)
            v2 = integrate.quad(y1, 0, T_obs
                                )[0]
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            n1 = n2
            # print(n1)
        h = n1
        return h

    delta_h1_theta_l = find_delta_h1_theta_l()

    def find_delta_h1_phi_l():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                              psi_s(theta_l, phi_l + delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l, phi_l + delta, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                 psi_s(theta_l, phi_l - delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l, phi_l - delta, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        n1 = 0.01
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, n1) * par_1(x, n1)
            v1 = integrate.quad(y, 0, T_obs
                                )[0]
            n2 = 0.1 * n1
            y1 = lambda x: par_1(x, n2) * par_1(x, n2)
            v2 = integrate.quad(y1, 0, T_obs
                                )[0]
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            n1 = n2
            # print(n1)
        h = n1
        return h

    delta_h1_phi_l = find_delta_h1_phi_l()

    def find_delta_h2_theta_s():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s_t + delta, phi_s_t(x),
                                              psi_s(theta_l, phi_l, theta_s + delta, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s_t + delta, phi_s_t(x),
                                  psi_s(theta_l, phi_l, theta_s + delta, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s_t - delta, phi_s_t(x),
                                 psi_s(theta_l, phi_l, theta_s - delta, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s_t - delta, phi_s_t(x),
                                  psi_s(theta_l, phi_l, theta_s - delta, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        n1 = 0.01
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, n1) * par_1(x, n1)
            v1 = integrate.quad(y, 0, T_obs
                                )[0]
            n2 = 0.1 * n1
            y1 = lambda x: par_1(x, n2) * par_1(x, n2)
            v2 = integrate.quad(y1, 0, T_obs
                                )[0]
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            n1 = n2
            # print(n1)
        h = n1
        return h

    delta_h2_theta_s = find_delta_h2_theta_s()

    def find_delta_h2_phi_s():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s_t, phi_s_t(x) + delta,
                                              psi_s(theta_l, phi_l, theta_s, phi_s + delta)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s_t, phi_s_t(x) + delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s + delta)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s_t, phi_s_t(x) - delta,
                                 psi_s(theta_l, phi_l, theta_s, phi_s - delta)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s_t, phi_s_t(x) - delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s - delta)) + fiD(
                    x))) / (
                    2 * delta)

        n1 = 0.01
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, n1) * par_1(x, n1)
            v1 = integrate.quad(y, 0, T_obs
                                )[0]
            n2 = 0.1 * n1
            y1 = lambda x: par_1(x, n2) * par_1(x, n2)
            v2 = integrate.quad(y1, 0, T_obs
                                )[0]
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            n1 = n2
            # print(n1)
        h = n1
        return h

    delta_h2_phi_s = find_delta_h2_phi_s()

    def find_delta_h2_theta_l():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                              psi_s(theta_l + delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l + delta, phi_l, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                 psi_s(theta_l - delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l - delta, phi_l, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        n1 = 0.01
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, n1) * par_1(x, n1)
            v1 = integrate.quad(y, 0, T_obs
                                )[0]
            n2 = 0.1 * n1
            y1 = lambda x: par_1(x, n2) * par_1(x, n2)
            v2 = integrate.quad(y1, 0, T_obs
                                )[0]
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            n1 = n2
            # print(n1)
        h = n1
        return h

    delta_h2_theta_l = find_delta_h2_theta_l()

    def find_delta_h2_phi_l():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                              psi_s(theta_l, phi_l + delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l, phi_l + delta, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                 psi_s(theta_l, phi_l - delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l, phi_l - delta, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        n1 = 0.01
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, n1) * par_1(x, n1)
            v1 = integrate.quad(y, 0, T_obs
                                )[0]
            n2 = 0.1 * n1
            y1 = lambda x: par_1(x, n2) * par_1(x, n2)
            v2 = integrate.quad(y1, 0, T_obs
                                )[0]
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            n1 = n2
            # print(n1)
        h = n1
        return h

    delta_h2_phi_l = find_delta_h2_phi_l()

    # 求偏导---------------------------------------------------------------------------------------------------------------
    def partial_h1_lnA(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.cos(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t))

    def partial_h1_psi_0(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t))

    def partial_h1_f(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s, )) * math.sin(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t)) * (
                2 * math.pi * t - p_P / c * K * math.sin(fi_t(t)))

    def partial_h1_f_dao(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t)) * (
                math.pow(t, 2) * math.pi - p_P * t / c * K * math.sin(fi_t(t)) - math.pow(p_P,
                                                                                          2) * K / (
                        2 * math.pi * c) * math.cos(fi_t(t)))

    def partial_h1_theta_s(t):
        h = delta_h1_theta_s

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s_t + delta, phi_s_t(x),
                                              psi_s(theta_l, phi_l, theta_s + delta, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s_t + delta, phi_s_t(x),
                                  psi_s(theta_l, phi_l, theta_s + delta, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s_t - delta, phi_s_t(x),
                                 psi_s(theta_l, phi_l, theta_s - delta, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s_t - delta, phi_s_t(x),
                                  psi_s(theta_l, phi_l, theta_s - delta, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h1_phi_s(t):
        h = delta_h1_phi_s

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s_t, phi_s_t(x) + delta,
                                              psi_s(theta_l, phi_l, theta_s, phi_s + delta)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s_t, phi_s_t(x) + delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s + delta)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s_t, phi_s_t(x) - delta,
                                 psi_s(theta_l, phi_l, theta_s, phi_s - delta)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s_t, phi_s_t(x) - delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s - delta)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h1_theta_l(t):
        h = delta_h1_theta_l

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                              psi_s(theta_l + delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l + delta, phi_l, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                 psi_s(theta_l - delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l - delta, phi_l, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h1_phi_l(t):
        h = delta_h1_phi_l

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                              psi_s(theta_l, phi_l + delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l, phi_l + delta, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                 psi_s(theta_l, phi_l - delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l, phi_l - delta, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h1_K(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(
                t)) * (-1) * p_P * f / c * math.sin(
            fi_t(t))

    def partial_h1_P(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t)) * (
                (-1) * f * K / c * math.sin(fi_t(t)) + 2 * math.pi * f * t / (
                c * p_P) * K * math.cos(fi_t(t)))

    def partial_h1_phi_0(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(
                t)) * ((-1) * p_P * f * K / c * math.cos(fi_t(t)) - 2 * math.pi * f_obs(t) * R_earth * math.sin(
            theta_s) * math.cos(phi_0 + 2 * math.pi * t / P_earth - phi_s) / c)

    # h2的偏导——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    def partial_h2_lnA(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.cos(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t))

    def partial_h2_psi_0(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.cos(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t))

    def partial_h2_f(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t)) * (
                2 * math.pi * t - p_P / c * K * math.sin(fi_t(t)))

    def partial_h2_f_dao(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t)) * (
                math.pow(t, 2) * math.pi - p_P * t / c * K * math.sin(fi_t(t)) - math.pow(p_P,
                                                                                          2) * K / (
                        2 * math.pi * c) * math.cos(fi_t(t)))

    def partial_h2_theta_s(t):
        h = delta_h2_theta_s

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s_t + delta, phi_s_t(x),
                                              psi_s(theta_l, phi_l, theta_s + delta, phi_s)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s_t + delta, phi_s_t(x),
                                  psi_s(theta_l, phi_l, theta_s + delta, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s_t - delta, phi_s_t(x),
                                 psi_s(theta_l, phi_l, theta_s - delta, phi_s)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s_t - delta, phi_s_t(x),
                                  psi_s(theta_l, phi_l, theta_s - delta, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h2_phi_s(t):
        h = delta_h2_phi_s

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s_t, phi_s_t(x) + delta,
                                              psi_s(theta_l, phi_l, theta_s, phi_s + delta)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s_t, phi_s_t(x) + delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s + delta)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s_t, phi_s_t(x) - delta,
                                 psi_s(theta_l, phi_l, theta_s, phi_s - delta)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s_t, phi_s_t(x) - delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s - delta)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h2_theta_l(t):
        h = delta_h2_theta_l

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                              psi_s(theta_l + delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l + delta, phi_l, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                 psi_s(theta_l - delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l - delta, phi_l, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h2_phi_l(t):
        h = delta_h2_phi_l

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                              psi_s(theta_l, phi_l + delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l, phi_l + delta, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                 psi_s(theta_l, phi_l - delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s_t, phi_s_t(x),
                                  psi_s(theta_l, phi_l - delta, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h2_K(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(
                t)) * (-1) * p_P * f / c * math.sin(
            fi_t(t))

    def partial_h2_P(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t)) * (
                (-1) * f * K / c * math.sin(fi_t(t)) + 2 * math.pi * f * t / (
                c * p_P) * K * math.cos(
            fi_t(t)))

    def partial_h2_phi_0(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s_t, phi_s_t(t),
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(
                t)) * ((-1) * p_P * f * K / c * math.cos(fi_t(t)) - 2 * math.pi * f_obs(t) * R_earth * math.sin(
            theta_s) * math.cos(
            phi_0 + 2 * math.pi * t / P_earth - phi_s) / c)

    # 将偏导映射到字典里
    X = dict()
    Y = dict()
    partial_h1 = []
    partial_h2 = []

    def VECTOR():
        X['partial_h1_lnA'] = partial_h1_lnA
        X['partial_h1_psi_0'] = partial_h1_psi_0
        X['partial_h1_f'] = partial_h1_f
        X['partial_h1_f_dao'] = partial_h1_f_dao
        X['partial_h1_theta_s'] = partial_h1_theta_s
        X['partial_h1_phi_s'] = partial_h1_phi_s
        X['partial_h1_theta_l'] = partial_h1_theta_l
        X['partial_h1_phi_l'] = partial_h1_phi_l
        X['partial_h1_K'] = partial_h1_K
        X['partial_h1_P'] = partial_h1_P
        X['partial_h1_phi_0'] = partial_h1_phi_0

        Y['partial_h2_lnA'] = partial_h2_lnA
        Y['partial_h2_psi_0'] = partial_h2_psi_0
        Y['partial_h2_f'] = partial_h2_f
        Y['partial_h2_f_dao'] = partial_h2_f_dao
        Y['partial_h2_theta_s'] = partial_h2_theta_s
        Y['partial_h2_phi_s'] = partial_h2_phi_s
        Y['partial_h2_theta_l'] = partial_h2_theta_l
        Y['partial_h2_phi_l'] = partial_h2_phi_l
        Y['partial_h2_K'] = partial_h2_K
        Y['partial_h2_P'] = partial_h2_P
        Y['partial_h2_phi_0'] = partial_h2_phi_0

        for key in X:
            partial_h1.append(X[key])
        for key in Y:
            partial_h2.append(Y[key])

        return 0

    VECTOR()

    FF = np.empty((len(partial_h1), len(partial_h2)))
    for k in range(len(partial_h1)):
        for n in range(len(partial_h2)):
            y = lambda t: partial_h1[k](t) * partial_h1[n](t) + partial_h2[k](t) * partial_h2[n](t)
            v = integrate.quad(y, 0, T_obs)
            FF[k][n] = v[0] * 2 * p_S_n_ni

    return FF
