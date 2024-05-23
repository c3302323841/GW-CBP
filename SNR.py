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
# S_n_ni = 1 / S_n_tianqin
# S_n_ni = 1 / S_n_lisa
S_n_ni = 1 / S_n_taiji
# S_n_ni = 1 / S_n_tianqin + 1 / S_n_taiji
# S_n_ni = 1 / S_n_tianqin + 1 / S_n_lisa
# S_n_ni = 1 / S_n_taiji + 1 / S_n_lisa
# S_n_ni = 1 / S_n_taiji + 1 / S_n_taiji + 1 / S_n_lisa
# ----------------------------------------------------------------------------------------------------------------------
# 无量纲常数
i = math.pi / 3  # 康亚城设定的常数,源轨道倾角（随机分布）
psi_0 = 0  # rad 康亚城设定的常数,观测者初相位
phi_0 = 0  # 源轨道初相位

theta_s_h = 1.27  # 源的坐标
phi_s_h = 5  # 源的坐标
theta_l = 1.77  # 轨道坐标 # tamanni参数
phi_l = 4  # 轨道坐标 # tamanni参数

M_b = 1 * M_sun
MM = math.pow(0.25 * M_b * M_b, 3 / 5) / math.pow(M_b, 0.2)  # 啁啾质量
f_dao = 96 / 5 * math.pow(G * MM / math.pow(c, 3), 5 / 3) * math.pow(math.pi, 8 / 3) * math.pow(f, 11 / 3)
P_x = []  # x数组
R_x = []  # x数组
SNR_q = []
na = 50  # 周期x格点数
nb = 1000  # 质量y格点数
for a in range(na):  # 循环周期
    P = math.pow(10, a * 0.2 - 1.6) * year  # 周期x格点数na要奇数才能通过0
    P_x.append(a * 0.2 - 1.6)
    M_p = math.pow(10, (50 / 100 * 3 - 1)) * M_J



    def psi_s(t, p_theta_s_h, p_phi_s_h, p_theta_l, p_phi_l):
        # return math.atan((0.5 * math.cos(p_theta_l) - math.sqrt(3) / 2 * math.sin(p_theta_l) * math.cos(
        #     2 * math.pi * t / P + phi_0 - p_phi_l)
        #                   - (math.cos(p_theta_l) * math.cos(p_theta_s_h) - math.sin(p_theta_l) * math.sin(
        #             p_theta_s_h) * math.cos(p_phi_l - p_phi_s_h))) /
        #                  (0.5 * math.sin(p_theta_l) * math.sin(p_theta_s_h) * math.sin(p_phi_l - p_phi_s_h) -
        #                   math.sqrt(3) / 2 * math.cos(2 * math.pi * t / P + phi_0) * (
        #                           math.cos(p_theta_l) * math.sin(p_theta_s_h) * math.sin(p_phi_s_h) - math.cos(
        #                       p_theta_s_h) * math.sin(p_theta_l) * math.sin(p_phi_l)) -
        #                   math.sqrt(3) / 2 * math.cos(2 * math.pi * t / P + phi_0) * (
        #                           math.cos(p_theta_s_h) * math.sin(p_theta_l) * math.cos(p_phi_l) - math.cos(
        #                       p_theta_l) * math.sin(p_theta_s_h) * math.sin(p_phi_s_h))))
        return 0.3992659778496011


    def i(p_theta_l, p_phi_l, p_theta_s, p_phi_s):
        # return math.cos(p_theta_l) * math.cos(p_theta_s) + math.sin(p_theta_l) * math.sin(
        #     p_theta_s) * math.cos(p_phi_l - p_phi_s)
        return math.pi / 3


    R = math.pow(G * P ** 2 * (M_b + M_p) / (4 * math.pi ** 2), 1 / 3)


    def K(p_i):
        return math.pow(2 * math.pi * G / P, 1 / 3) * M_p / math.pow(M_p + M_b, 2 / 3) * math.sin(
            p_i)


    def theta_s(t, p_theta_s_h, p_phi_s_h):
        return math.acos(
            0.5 * math.cos(p_theta_s_h) - 0.5 * pow(3, 0.5) * math.sin(p_theta_s_h) * math.cos(fi_t(t) - p_phi_s_h))


    def phi_s(t, p_theta_s_h, p_phi_s_h):
        return 2 * math.pi * t / P_earth - math.atan(
            (pow(3, 0.5) * math.cos(p_theta_s_h) + math.sin(p_theta_s_h) * math.cos(fi_t(t) - p_phi_s_h)) / (
                    2 * math.sin(p_theta_s_h) * math.cos(fi_t(t) - p_phi_s_h)))


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
            -1 * A_cha(p_i) * F1_cha(p_theta_s, p_phi_s, p_psi_s) / A_jia(p_i) / F1_jia(p_theta_s, p_phi_s, p_psi_s))


    def fi_2(p_i, p_theta_s, p_phi_s, p_psi_s):
        return math.atan(
            -1 * A_cha(p_i) * F2_cha(p_theta_s, p_phi_s, p_psi_s) / A_jia(p_i) / F2_jia(p_theta_s, p_phi_s, p_psi_s))


    def fi_t(t):
        return 2 * math.pi * t / P_earth + phi_0


    def f_obs(t, p_i):
        return (1 - K(p_i) * math.cos(2 * math.pi * t / P + phi_0) / c) * (f + f_dao * t)


    def fiD(t, p_i):
        return 2 * math.pi * f_obs(t, p_i) * R_earth * math.sin(theta_s_h) * math.cos(
            phi_0 + 2 * math.pi * t / P_earth - phi_s_h) / c


    def psi_obs(t, p_i):
        return 2 * math.pi * (f + 0.5 * f_dao * t) * t - P * f * K(p_i) / c * math.sin(
            fi_t(t)) - P * f_dao * t / c * K(p_i) * math.sin(fi_t(t)) - P * P * f_dao * K(p_i) / (
                    2 * math.pi * c) * math.cos(
            fi_t(t))


    def h1(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.cos(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_1(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l, theta_s_h,
                                                                                              phi_s_h)))


    def h2(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.cos(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_2(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l, theta_s_h,
                                                                                              phi_s_h)))


    snr = lambda t: 2 * (h1(t) * h1(t) + h2(t) * h2(t)) * S_n_ni
    # snr = lambda t: 2 * (h1(t) * h1(t) + h2(t) * h2(t))
    SNR = integrate.quad(snr, 0, T_obs)
    SNR_q.append(pow(SNR[0], 0.5))

plt.plot(P_x, SNR_q, marker='o')
# plt.xlabel('distance to the binary star [AU] log')
plt.xlabel('periodic of exoplanet  [year] log')
plt.ylabel('SNR')
plt.title('taiji 5 mHz T = 1 year Mb = 1 Msun')
plt.axvline(x=0, linestyle=":", color="m")
plt.show(block=True)

print(P_x, SNR_q)

# # 准备数据
# dataR_x = pd.DataFrame(R_x)  # 关键1，将ndarray格式转换为DataFrame
# dataM_p_y = pd.DataFrame(M_p_y)
#
# writer = pd.ExcelWriter('taiji.xlsx')  # 关键2，创建名称为hhh的excel表格
# dataR_x.to_excel(writer, 'page_1', float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。若多个文件，可以在page_2中写入
# dataM_p_y.to_excel(writer, 'page_2', float_format='%.5f')
# writer.save()
