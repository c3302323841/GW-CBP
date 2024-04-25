import math
import numpy as np
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib
# import pandas as pd

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

# 单位
# kg = 1
# M_J_dim = 1.898e27 * kg
# s = 1
# year = 31536000 * s
# month = year / 12
# day = month / 30
# m = 1
# AU = 1.496e11 * m
# pc = 30835997962819660.8 * m
# M_dimension = 1 / kg
# T_dimension = 1 / s
# L_dimension = 1 / m
# 常量
M_J = 1.898e27 * M_dimension  # 木星质量
M_sun = 2.0e30 * M_dimension  # 太阳质量
G = 6.67e-11 * L_dimension ** 3 / (M_dimension * T_dimension ** 2)  # m^3/(kg * s^2)
c = 3e8 * L_dimension / T_dimension  # m/s
d = 2062650000 * 1.496e11 * L_dimension  # 源光度距离
R_earth = 1 * 1.496e11 * L_dimension  # 地球半径
P_earth = 1 * 31536000 * T_dimension  # 地球公转周期
# f = 0.01 / T_dimension  # 引力波频率（10 mHz）s-1
f = 0.01 / T_dimension  # ######################################################################################################################
T_obs = 4 * 31536000 * T_dimension  # 观测周期
# 功率谱密度
# tianqin
L_TianQin = math.pow(3, 0.5) * 1e8 * L_dimension  # m
f_xing_TianQin = c / (2 * math.pi * L_TianQin)
S_n_tianqin = 10 / 3 / L_TianQin ** 2 * (
        (1e-24 * L_dimension ** 2 * T_dimension) + 4 * (1e-30 * L_dimension ** 2 / T_dimension ** 3) / (
        2 * math.pi * f / T_dimension) ** 4 * (1 + 1e-4 / f)) * (1 + 6 / 10 * (f / f_xing_TianQin) ** 2)
# lisa
P_OMS_lisa = 1.5e-11 ** 2 * (1 + (0.002 / T_dimension / f) ** 4) * L_dimension ** 2 * T_dimension
P_acc_lisa = 3e-15 ** 2 * (1 + (0.0004 / T_dimension / f) ** 2) * (
        1 + (f / 0.008 / T_dimension) ** 4) * L_dimension ** 2 / T_dimension ** 3
L_lisa = 2.5e9 * L_dimension
f_xing_lisa = 0.01909 / T_dimension
S_n_lisa = 10 / 3 / L_lisa ** 2 * (
        P_OMS_lisa + 2 * (1 + (math.cos(f / f_xing_lisa)) ** 2) * P_acc_lisa / (2 * math.pi * f) ** 4) * (
                            1 + 6 / 10 * (f * f_xing_lisa) ** 2)
# taiji
P_OMS_tj = 8e-12 ** 2 * (1 + (0.002 / T_dimension / f) ** 4) * L_dimension ** 2 * T_dimension
P_acc_tj = 3e-15 ** 2 * (1 + (0.0004 / T_dimension / f) ** 2) * (
        1 + (f / 0.008 / T_dimension) ** 4) * L_dimension ** 2 / T_dimension ** 3
L_tj = 3e9 * L_dimension
f_xing_tj = c / (2 * math.pi * L_tj)
S_n_taiji = 10 / 3 / L_tj ** 2 * (P_OMS_tj + 2 * (1 + (math.cos(f / f_xing_tj)) ** 2) * P_acc_tj / (2 * math.pi * f) ** 4) * (
            1 + 6 / 10 * (f * f_xing_tj) ** 2)
# ----------------------------------------------------------------------------------------------------------------------
S_n_ni = 1 / S_n_tianqin
# S_n_ni = 1 / S_n_lisa
# S_n_ni = 1 / S_n_taiji
# S_n_ni = 1 / S_n_tianqin + 1 / S_n_taiji
# S_n_ni = 1 / S_n_tianqin + 1 / S_n_lisa
# S_n_ni = 1 / S_n_taiji + 1 / S_n_lisa
# S_n_ni = 1 / S_n_taiji + 1 / S_n_taiji + 1 / S_n_lisa
# ----------------------------------------------------------------------------------------------------------------------
# 无量纲常数
i = math.pi / 3  # 康亚城设定的常数,源轨道倾角（随机分布）
psi_0 = 0  # rad 康亚城设定的常数,观测者初相位
phi_0 = 0  # 源轨道初相位

theta_s = math.acos(0.3)  # 源的坐标
phi_s = 5  # 源的坐标
theta_l = math.acos(-0.2)  # 轨道坐标 # tamanni参数
phi_l = 4  # 轨道坐标 # tamanni参数

M_b = 1 * M_sun
MM = math.pow(0.25 * M_b * M_b, 3 / 5) / math.pow(M_b, 0.2)  # 啁啾质量
f_dao = 96 / 5 * math.pow(G * MM / math.pow(c, 3), 5 / 3) * math.pow(math.pi, 8 / 3) * math.pow(f, 11 / 3)
P_x = []  # x数组
R_x = []  # x数组
M_p_y = []  # y数组
SNR_q = []
na = 20  # 周期x格点数
nb = 100  # 质量y格点数
for a in range(na):  # 循环周期
    # xlist = [-1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1]
    # P = math.pow(10, xlist[a]) * year
    P = math.pow(10, a * 0.2 - 1.6) * year  # 周期x格点数na要奇数才能通过0

    # 误差图坐标列表
    b_x = []
    Re_y = []
    Re_z = []
    for b in range(nb):  # 循环质量
        print(a, b)
        M_p = math.pow(10, (b / nb * 3 - 1)) * M_J
        # M_p = math.pow(10, (0.5 * b)) * M_J  # 最初成功代码，由于步长太大精度低舍弃
        R = math.pow(G * P ** 2 * (M_b + M_p) / (4 * math.pi ** 2), 1 / 3)
        K = math.pow(2 * math.pi * G / P, 1 / 3) * M_p / math.pow(M_p + M_b, 2 / 3) * math.sin(i)

        # 定义函数以获得h1,h2-------------------------------------------------------------------------------------------------
        def psi_s(t, p_theta_l, p_phi_l, p_theta_s, p_phi_s, p_phi_0):
            return math.atan((0.5 * math.cos(p_theta_l) - math.sqrt(3) / 2 * math.sin(p_theta_l) * math.cos(
                2 * math.pi * t / P + p_phi_0 - p_phi_l)
                              - (math.cos(p_theta_l) * math.cos(p_theta_s) + math.sin(p_theta_l) * math.sin(
                        p_theta_s) * math.cos(p_phi_l - p_phi_s))
                              * (math.sin(p_theta_s) * math.cos(p_phi_s) + math.sin(p_theta_s) * math.sin(
                        p_phi_s) + math.cos(p_phi_s))) /
                             (0.5 * math.sin(p_theta_l) * math.sin(p_theta_s) * math.cos(p_phi_l - p_phi_s) -
                              math.sqrt(3) / 2 * math.cos(2 * math.pi * t / P + p_phi_0) * (
                                      math.cos(p_theta_l) * math.sin(p_theta_s) * math.sin(p_phi_s) - math.cos(
                                  p_theta_s) * math.sin(p_theta_l) * math.sin(p_phi_l)) -
                              math.sqrt(3) / 2 * math.cos(2 * math.pi * t / P + p_phi_0) * (
                                      math.cos(p_theta_s) * math.sin(p_theta_l) * math.sin(p_phi_l) - math.cos(
                                  p_theta_l) * math.sin(p_theta_s) * math.sin(p_phi_s))))

        # 康亚成参数
        # theta_l = 6.06685762621992 - math.pi  # 轨道坐标
        # phi_l = 10.2163276809597 - 2 * math.pi  # 轨道坐标
        # def psi_s(t, p_theta_l, p_phi_l, p_theta_s, p_phi_s, p_phi_0):
        #     return psi_0
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


        AA = 2 * math.pow(G * MM, 5 / 3) * math.pow(math.pi * f, 2 / 3) / (math.pow(c, 4) * d)
        A_jia = AA * (1 + math.pow(math.cos(i), 2))
        A_cha = 2 * AA * math.cos(i)


        def A1(p_theta_s, p_phi_s, p_psi_s):
            return math.pow(
                math.pow(A_jia, 2) * math.pow(F1_jia(p_theta_s, p_phi_s, p_psi_s), 2) + math.pow(A_cha, 2) * math.pow(
                    F1_cha(p_theta_s, p_phi_s, p_psi_s), 2), 1 / 2)


        def A2(p_theta_s, p_phi_s, p_psi_s):
            return math.pow(
                math.pow(A_jia, 2) * math.pow(F2_jia(p_theta_s, p_phi_s, p_psi_s), 2) + math.pow(A_cha, 2) * math.pow(
                    F2_cha(p_theta_s, p_phi_s, p_psi_s), 2), 1 / 2)


        def fi_1(p_theta_s, p_phi_s, p_psi_s):
            return math.atan(
                -1 * A_cha * F1_cha(p_theta_s, p_phi_s, p_psi_s) / A_jia / F1_jia(p_theta_s, p_phi_s, p_psi_s))


        def fi_2(p_theta_s, p_phi_s, p_psi_s):
            return math.atan(
                -1 * A_cha * F2_cha(p_theta_s, p_phi_s, p_psi_s) / A_jia / F2_jia(p_theta_s, p_phi_s, p_psi_s))


        def fi_t(t):
            return 2 * math.pi * t / P + phi_0


        def f_obs(t):
            return (1 - K * math.cos(2 * math.pi * t / P + phi_0) / c) * (f + f_dao * t)


        def fiD(t):
            return 2 * math.pi * f_obs(t) * R_earth * math.sin(theta_s) * math.cos(
                phi_0 + 2 * math.pi * t / P_earth - phi_s) / c


        def psi_obs(t):
            return 2 * math.pi * (f + 0.5 * f_dao * t) * t - P * f * K / c * math.sin(
                fi_t(t)) - P * f_dao * t / c * K * math.sin(fi_t(t)) - P * P * f_dao * K / (2 * math.pi * c) * math.cos(
                fi_t(t))


        def h1(t):
            return math.pow(3, 0.5) / 2 * A1(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) * math.cos(
                psi_obs(t) + fi_1(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(t))


        def h2(t):
            return math.pow(3, 0.5) / 2 * A1(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) * math.cos(
                psi_obs(t) + fi_2(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(t))


        def find_delta_h1_theta_s():
            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A1(theta_s + delta, phi_s,
                                                  psi_s(x, theta_l, phi_l, theta_s + delta, phi_s,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s + delta, phi_s,
                                      psi_s(x, theta_l, phi_l, theta_s + delta, phi_s, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A1(theta_s - delta, phi_s,
                                     psi_s(x, theta_l, phi_l, theta_s - delta, phi_s, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s - delta, phi_s,
                                      psi_s(x, theta_l, phi_l, theta_s - delta, phi_s, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            h = 1e-7 * theta_s
            step = 0
            while step < 5:
                step += 1
                y = lambda x: par_1(x, h) * par_1(x, h)
                v1 = integrate.quad(y, 0, T_obs)[0] * 2 * S_n_ni
                h = pow(v1, -0.5) / 10
            return h


        delta_h1_theta_s = find_delta_h1_theta_s()


        def find_delta_h1_phi_s():
            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A1(theta_s, phi_s + delta,
                                                  psi_s(x, theta_l, phi_l, theta_s, phi_s + delta,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s + delta,
                                      psi_s(x, theta_l, phi_l, theta_s, phi_s + delta, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A1(theta_s, phi_s - delta,
                                     psi_s(x, theta_l, phi_l, theta_s, phi_s - delta, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s - delta,
                                      psi_s(x, theta_l, phi_l, theta_s, phi_s - delta, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            h = 1e-7 * phi_s
            step = 0
            while step < 5:
                step += 1
                y = lambda x: par_1(x, h) * par_1(x, h)
                v1 = integrate.quad(y, 0, T_obs)[0] * 2 * S_n_ni
                h = pow(v1, -0.5) / 10
            return h


        delta_h1_phi_s = find_delta_h1_phi_s()


        def find_delta_h1_theta_l():
            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A1(theta_s, phi_s,
                                                  psi_s(x, theta_l + delta, phi_l, theta_s, phi_s,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l + delta, phi_l, theta_s, phi_s, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A1(theta_s, phi_s,
                                     psi_s(x, theta_l - delta, phi_l, theta_s, phi_s, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l - delta, phi_l, theta_s, phi_s, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            h = 1e-7 * theta_l
            step = 0
            while step < 5:
                step += 1
                y = lambda x: par_1(x, h) * par_1(x, h)
                v1 = integrate.quad(y, 0, T_obs)[0] * 2 * S_n_ni
                h = pow(v1, -0.5) / 10
            return h


        delta_h1_theta_l = find_delta_h1_theta_l()


        def find_delta_h1_phi_l():
            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A1(theta_s, phi_s,
                                                  psi_s(x, theta_l, phi_l + delta, theta_s, phi_s,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l, phi_l + delta, theta_s, phi_s, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A1(theta_s, phi_s,
                                     psi_s(x, theta_l, phi_l - delta, theta_s, phi_s, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l, phi_l - delta, theta_s, phi_s, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            h = 1e-7 * phi_l
            step = 0
            while step < 5:
                step += 1
                y = lambda x: par_1(x, h) * par_1(x, h)
                v1 = integrate.quad(y, 0, T_obs)[0] * 2 * S_n_ni
                h = pow(v1, -0.5) / 10
            return h


        delta_h1_phi_l = find_delta_h1_phi_l()


        def find_delta_h2_theta_s():
            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A2(theta_s + delta, phi_s,
                                                  psi_s(x, theta_l, phi_l, theta_s + delta, phi_s,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s + delta, phi_s,
                                      psi_s(x, theta_l, phi_l, theta_s + delta, phi_s, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A2(theta_s - delta, phi_s,
                                     psi_s(x, theta_l, phi_l, theta_s - delta, phi_s, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s - delta, phi_s,
                                      psi_s(x, theta_l, phi_l, theta_s - delta, phi_s, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            h = 1e-7 * theta_s
            step = 0
            while step < 5:
                step += 1
                y = lambda x: par_1(x, h) * par_1(x, h)
                v1 = integrate.quad(y, 0, T_obs)[0] * 2 * S_n_ni
                h = pow(v1, -0.5) / 10
            return h


        delta_h2_theta_s = find_delta_h2_theta_s()


        def find_delta_h2_phi_s():
            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A2(theta_s, phi_s + delta,
                                                  psi_s(x, theta_l, phi_l, theta_s, phi_s + delta,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s + delta,
                                      psi_s(x, theta_l, phi_l, theta_s, phi_s + delta, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A2(theta_s, phi_s - delta,
                                     psi_s(x, theta_l, phi_l, theta_s, phi_s - delta, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s - delta,
                                      psi_s(x, theta_l, phi_l, theta_s, phi_s - delta, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            h = 1e-7 * phi_s
            step = 0
            while step < 5:
                step += 1
                y = lambda x: par_1(x, h) * par_1(x, h)
                v1 = integrate.quad(y, 0, T_obs)[0] * 2 * S_n_ni
                h = pow(v1, -0.5) / 10
            return h


        delta_h2_phi_s = find_delta_h2_phi_s()


        def find_delta_h2_theta_l():
            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A2(theta_s, phi_s,
                                                  psi_s(x, theta_l + delta, phi_l, theta_s, phi_s,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l + delta, phi_l, theta_s, phi_s, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A2(theta_s, phi_s,
                                     psi_s(x, theta_l - delta, phi_l, theta_s, phi_s, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l - delta, phi_l, theta_s, phi_s, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            h = 1e-7 * theta_l
            step = 0
            while step < 5:
                step += 1
                y = lambda x: par_1(x, h) * par_1(x, h)
                v1 = integrate.quad(y, 0, T_obs)[0] * 2 * S_n_ni
                h = pow(v1, -0.5) / 10
            return h


        delta_h2_theta_l = find_delta_h2_theta_l()


        def find_delta_h2_phi_l():
            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A2(theta_s, phi_s,
                                                  psi_s(x, theta_l, phi_l + delta, theta_s, phi_s,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l, phi_l + delta, theta_s, phi_s, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A2(theta_s, phi_s,
                                     psi_s(x, theta_l, phi_l - delta, theta_s, phi_s, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l, phi_l - delta, theta_s, phi_s, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            h = 1e-1 * phi_l
            step = 0
            while step < 5:
                step += 1
                y = lambda x: par_1(x, h) * par_1(x, h)
                v1 = integrate.quad(y, 0, T_obs)[0] * 2 * S_n_ni
                h = pow(v1, -0.5) / 10
            return h


        delta_h2_phi_l = find_delta_h2_phi_l()

        # 求偏导---------------------------------------------------------------------------------------------------------------
        def partial_h1_lnA(t):
            return math.pow(3, 0.5) / 2 * A1(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s,
                                                                   phi_0)) * math.cos(
                psi_obs(t) + fi_1(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(t))


        def partial_h1_psi_0(t):
            return math.pow(3, 0.5) / 2 * A1(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s,
                                                   phi_0)) * math.sin(
                psi_obs(t) + fi_1(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(t))


        def partial_h1_f(t):
            return math.pow(3, 0.5) / 2 * A1(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) * math.sin(
                psi_obs(t) + fi_2(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(t)) * (
                    2 * math.pi * t - P / c * K * math.sin(fi_t(t)))


        def partial_h1_f_dao(t):
            return math.pow(3, 0.5) / 2 * A1(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s,
                                                   phi_0)) * math.sin(
                psi_obs(t) + fi_2(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(t)) * (
                    math.pow(t, 2) * math.pi - P * t / c * K * math.sin(fi_t(t)) - math.pow(P,
                                                                                            2) * K / (
                            2 * math.pi * c) * math.cos(fi_t(t)))


        def partial_h1_theta_s(t):
            h = delta_h1_theta_s

            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A1(theta_s + delta, phi_s,
                                                  psi_s(x, theta_l, phi_l, theta_s + delta, phi_s,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s + delta, phi_s,
                                      psi_s(x, theta_l, phi_l, theta_s + delta, phi_s, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A1(theta_s - delta, phi_s,
                                     psi_s(x, theta_l, phi_l, theta_s - delta, phi_s, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s - delta, phi_s,
                                      psi_s(x, theta_l, phi_l, theta_s - delta, phi_s, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            return par_1(t, h)


        def partial_h1_phi_s(t):
            h = delta_h1_phi_s

            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A1(theta_s, phi_s + delta,
                                                  psi_s(x, theta_l, phi_l, theta_s, phi_s + delta,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s + delta,
                                      psi_s(x, theta_l, phi_l, theta_s, phi_s + delta, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A1(theta_s, phi_s - delta,
                                     psi_s(x, theta_l, phi_l, theta_s, phi_s - delta, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s - delta,
                                      psi_s(x, theta_l, phi_l, theta_s, phi_s - delta, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            return par_1(t, h)


        def partial_h1_theta_l(t):
            h = delta_h1_theta_l

            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A1(theta_s, phi_s,
                                                  psi_s(x, theta_l + delta, phi_l, theta_s, phi_s,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l + delta, phi_l, theta_s, phi_s, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A1(theta_s, phi_s,
                                     psi_s(x, theta_l - delta, phi_l, theta_s, phi_s, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l - delta, phi_l, theta_s, phi_s, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            return par_1(t, h)


        def partial_h1_phi_l(t):
            h = delta_h1_phi_l

            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A1(theta_s, phi_s,
                                                  psi_s(x, theta_l, phi_l + delta, theta_s, phi_s,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l, phi_l + delta, theta_s, phi_s, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A1(theta_s, phi_s,
                                     psi_s(x, theta_l, phi_l - delta, theta_s, phi_s, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l, phi_l - delta, theta_s, phi_s, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            return par_1(t, h)


        def partial_h1_K(t):
            return math.pow(3, 0.5) / 2 * A1(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) * math.sin(
                psi_obs(t) + fi_1(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(
                    t)) * (-1) * P * f / c * math.sin(
                fi_t(t))


        def partial_h1_P(t):
            return math.pow(3, 0.5) / 2 * A1(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) * math.sin(
                psi_obs(t) + fi_1(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(t)) * (
                    (-1) * f * K / c * math.sin(fi_t(t)) + 2 * math.pi * f * t / (
                    c * P) * K * math.cos(fi_t(t)))


        def partial_h1_phi_0(t):
            return math.pow(3, 0.5) / 2 * A1(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) * math.sin(
                psi_obs(t) + fi_1(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(
                    t)) * ((-1) * P * f * K / c * math.cos(fi_t(t)) - 2 * math.pi * f_obs(t) * R_earth * math.sin(
                theta_s) * math.cos(phi_0 + 2 * math.pi * t / P_earth - phi_s) / c)


        # h2的偏导——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
        def partial_h2_lnA(t):
            return math.pow(3, 0.5) / 2 * A2(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s,
                                                   phi_0)) * math.cos(
                psi_obs(t) + fi_2(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(t))


        def partial_h2_psi_0(t):
            return math.pow(3, 0.5) / 2 * A2(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s,
                                                   phi_0)) * math.cos(
                psi_obs(t) + fi_2(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(t))


        def partial_h2_f(t):
            return math.pow(3, 0.5) / 2 * A2(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) * math.sin(
                psi_obs(t) + fi_2(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(t)) * (
                    2 * math.pi * t - P / c * K * math.sin(fi_t(t)))


        def partial_h2_f_dao(t):
            return math.pow(3, 0.5) / 2 * A2(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s,
                                                   phi_0)) * math.sin(
                psi_obs(t) + fi_2(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(t)) * (
                    math.pow(t, 2) * math.pi - P * t / c * K * math.sin(fi_t(t)) - math.pow(P,
                                                                                            2) * K / (
                            2 * math.pi * c) * math.cos(fi_t(t)))


        def partial_h2_theta_s(t):
            h = delta_h2_theta_s

            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A2(theta_s + delta, phi_s,
                                                  psi_s(x, theta_l, phi_l, theta_s + delta, phi_s,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s + delta, phi_s,
                                      psi_s(x, theta_l, phi_l, theta_s + delta, phi_s, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A2(theta_s - delta, phi_s,
                                     psi_s(x, theta_l, phi_l, theta_s - delta, phi_s, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s - delta, phi_s,
                                      psi_s(x, theta_l, phi_l, theta_s - delta, phi_s, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            return par_1(t, h)


        def partial_h2_phi_s(t):
            h = delta_h2_phi_s

            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A2(theta_s, phi_s + delta,
                                                  psi_s(x, theta_l, phi_l, theta_s, phi_s + delta,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s + delta,
                                      psi_s(x, theta_l, phi_l, theta_s, phi_s + delta, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A2(theta_s, phi_s - delta,
                                     psi_s(x, theta_l, phi_l, theta_s, phi_s - delta, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s - delta,
                                      psi_s(x, theta_l, phi_l, theta_s, phi_s - delta, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            return par_1(t, h)


        def partial_h2_theta_l(t):
            h = delta_h2_theta_l

            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A2(theta_s, phi_s,
                                                  psi_s(x, theta_l + delta, phi_l, theta_s, phi_s,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l + delta, phi_l, theta_s, phi_s, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A2(theta_s, phi_s,
                                     psi_s(x, theta_l - delta, phi_l, theta_s, phi_s, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l - delta, phi_l, theta_s, phi_s, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            return par_1(t, h)


        def partial_h2_phi_l(t):
            h = delta_h2_phi_l

            def par_1(x, delta):
                return (math.pow(3, 0.5) / 2 * A2(theta_s, phi_s,
                                                  psi_s(x, theta_l, phi_l + delta, theta_s, phi_s,
                                                        phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l, phi_l + delta, theta_s, phi_s, phi_0)) + fiD(
                        x)) - math.pow(
                    3, 0.5) / 2 * A2(theta_s, phi_s,
                                     psi_s(x, theta_l, phi_l - delta, theta_s, phi_s, phi_0)) * math.cos(
                    psi_obs(x) + fi_1(theta_s, phi_s,
                                      psi_s(x, theta_l, phi_l - delta, theta_s, phi_s, phi_0)) + fiD(
                        x))) / (
                        2 * delta)

            return par_1(t, h)


        def partial_h2_K(t):
            return math.pow(3, 0.5) / 2 * A2(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) * math.sin(
                psi_obs(t) + fi_2(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(
                    t)) * (-1) * P * f / c * math.sin(
                fi_t(t))


        def partial_h2_P(t):
            return math.pow(3, 0.5) / 2 * A2(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) * math.sin(
                psi_obs(t) + fi_2(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(t)) * (
                    (-1) * f * K / c * math.sin(fi_t(t)) + 2 * math.pi * f * t / (
                    c * P) * K * math.cos(
                fi_t(t)))


        def partial_h2_phi_0(t):
            return math.pow(3, 0.5) / 2 * A2(theta_s, phi_s,
                                             psi_s(t, theta_l, phi_l, theta_s, phi_s,
                                                   phi_0)) * math.sin(
                psi_obs(t) + fi_1(theta_s, phi_s, psi_s(t, theta_l, phi_l, theta_s, phi_s, phi_0)) + fiD(
                    t)) * ((-1) * P * f * K / c * math.cos(fi_t(t)) - 2 * math.pi * f_obs(t) * R_earth * math.sin(
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


        # 求K的误差
        def VAR_K():
            fisher = np.empty((len(partial_h1), len(partial_h2)))
            # 求fisher矩阵
            for k in range(len(partial_h1)):
                for n in range(len(partial_h2)):
                    y = lambda t: partial_h1[k](t) * partial_h1[n](t) + partial_h2[k](t) * partial_h2[n](t)
                    v = integrate.quad(y, 0, T_obs)
                    fisher[k][n] = v[0] * 2 * S_n_ni
            # fisher矩阵求逆
            # # QR求伪逆
            # def gram_schmidt(A):
            #     """Gram-schmidt正交化"""
            #     Q = np.zeros_like(A)
            #     cnt = 0
            #     for a in A.T:
            #         u = np.copy(a)
            #         for i in range(0, cnt):
            #             u -= np.dot(np.dot(Q[:, i].T, a), Q[:, i])  # 减去待求向量在以求向量上的投影
            #         e = u / np.linalg.norm(u)  # 归一化
            #         Q[:, cnt] = e
            #         cnt += 1
            #     R = np.dot(Q.T, A)
            #     return (Q, R)
            #
            # def givens_rotation(A):
            #     """Givens变换"""
            #     (r, c) = np.shape(A)
            #     Q = np.identity(r)
            #     R = np.copy(A)
            #     (rows, cols) = np.tril_indices(r, -1, c)
            #     for (row, col) in zip(rows, cols):
            #         if R[row, col] != 0:  # R[row, col]=0则c=1,s=0,R、Q不变
            #             r_ = np.hypot(R[col, col], R[row, col])  # d
            #             c = R[col, col] / r_
            #             s = -R[row, col] / r_
            #             G = np.identity(r)
            #             G[[col, row], [col, row]] = c
            #             G[row, col] = s
            #             G[col, row] = -s
            #             R = np.dot(G, R)  # R=G(n-1,n)*...*G(2n)*...*G(23,1n)*...*G(12)*A
            #             Q = np.dot(Q, G.T)  # Q=G(n-1,n).T*...*G(2n).T*...*G(23,1n).T*...*G(12).T
            #     return (Q, R)
            #
            # def householder_reflection(A):
            #     """Householder变换"""
            #     (r, c) = np.shape(A)
            #     Q = np.identity(r)
            #     R = np.copy(A)
            #     for cnt in range(r - 1):
            #         x = R[cnt:, cnt]
            #         e = np.zeros_like(x)
            #         e[0] = np.linalg.norm(x)
            #         u = x - e
            #         v = u / np.linalg.norm(u)
            #         Q_cnt = np.identity(r)
            #         Q_cnt[cnt:, cnt:] -= 2.0 * np.outer(v, v)
            #         R = np.dot(Q_cnt, R)  # R=H(n-1)*...*H(2)*H(1)*A
            #         Q = np.dot(Q, Q_cnt)  # Q=H(n-1)*...*H(2)*H(1)  H为自逆矩阵
            #     return (Q, R)
            #
            # (Q, R) = gram_schmidt(fisher)
            # (Q, R) = givens_rotation(fisher)
            # (Q, R) = householder_reflection(fisher)
            # print(Q)
            # print(R)
            # covariance = np.linalg.pinv(R) @ np.transpose(Q)

            # SVD求伪逆
            # u, s_s, v = linalg.svd(fisher)
            # ss = np.zeros((v.shape[0], u.shape[0]))
            # ss[:v.shape[0], :v.shape[0]] = np.diag(1 / s_s)
            # covariance = v.T.dot(ss).dot(u.T)

            covariance = linalg.inv(fisher)

            # 求逆出现负对角元，剔除之
            if covariance[8][8] < 0:
                var_K = K
                # var_K = math.pow(abs(covariance[8][8]), 1 / 2)
            else:
                var_K = math.pow(covariance[8][8], 1 / 2)  # variance 方差

            # 查看对角元和逆矩阵的特征值
            # for o in range(11):
            #     print(fisher[o][o])
            # print(linalg.eigvals(covariance))
            return var_K


        Re = VAR_K() / K  # relative error 相对误差
        # 画误差图
        b_x.append(b)
        Re_y.append(math.log10(Re))
        Re_z.append(Re)

        # 误差符合条件时加入质量周期图，并开始下一个周期计算
        if b > 0 and Re < 0.3 and Re_z[b] / Re_z[b - 1] > 0.5:
            snr = lambda t: 2 * (h1(t) * h1(t) + h2(t) * h2(t)) * S_n_ni
            SNR = integrate.quad(snr, 0, T_obs)
            SNR_q.append(SNR[0])
            x = np.array(Re_y)
            y = np.array(b_x)
            # 第一步：进行初步线性拟合
            coefficients_initial = np.polyfit(x, y, 1)
            fit_function_initial = np.poly1d(coefficients_initial)
            # 第二步：计算残差
            residuals = y - fit_function_initial(x)
            # 第三步：确定剔除标准
            mean_residual = np.mean(abs(residuals))
            std_residual = np.std(abs(residuals))
            threshold = mean_residual + std_residual
            # 第四步：剔除误差较大的数据点
            indices_to_keep = np.where(abs(residuals) < threshold)[0]
            x_filtered = x[indices_to_keep]
            y_filtered = y[indices_to_keep]
            # 第五步：使用剩余的数据点进行最终的线性拟合
            coefficients_final = np.polyfit(x_filtered, y_filtered, 1)
            fit_function_final = np.poly1d(coefficients_final)
            # 使用过滤后的模型进行预测
            y_predict = fit_function_final(math.log10(0.3))
            MP = math.log10(math.pow(10, (y_predict / nb * 3) - 1))
            P_x.append(math.log10(P / year))
            R_x.append(math.log10(R / AU))
            M_p_y.append(MP)
            break
    # 画误差图
    # plt.plot(b_x, Re_y, marker='o')
    # plt.title('taiji')
    # plt.show(block=True)
# 画质量周期图
plt.plot(P_x, M_p_y, marker='o')
# plt.xlabel('distance to the binary star [AU] log')
plt.xlabel('periodic of exoplanet  [year] log')
plt.ylabel('planetary mass [Mj] log')
plt.title('taiji 5 mHz T = 1 year Mb = 1 Msun')
plt.axvline(x=0, linestyle=":", color="m")
plt.show(block=True)

plt.plot(P_x, SNR_q, marker='o')
# plt.xlabel('distance to the binary star [AU] log')
plt.xlabel('periodic of exoplanet  [year] log')
plt.ylabel('SNR')
plt.title('taiji 5 mHz T = 1 year Mb = 1 Msun')
plt.axvline(x=0, linestyle=":", color="m")
plt.show(block=True)

print(P_x, M_p_y, SNR_q)

# # 准备数据
# dataR_x = pd.DataFrame(R_x)  # 关键1，将ndarray格式转换为DataFrame
# dataM_p_y = pd.DataFrame(M_p_y)
#
# writer = pd.ExcelWriter('taiji.xlsx')  # 关键2，创建名称为hhh的excel表格
# dataR_x.to_excel(writer, 'page_1', float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
# dataM_p_y.to_excel(writer, 'page_2', float_format='%.5f')
# writer.save()