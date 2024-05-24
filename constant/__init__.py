import math
import parameter as par

# 单位
kg_dim = 1
m_dim = 1
AU_dim = 1.496e11 * m_dim
pc_dim = 30835997962819660.8 * m_dim
s_dim = 1
year_dim = 31536000 * s_dim
month_dim = year_dim / 12
day_dim = month_dim / 30
# 单位选择
M_dimension = 1 / kg_dim
L_dimension = 1 / m_dim
T_dimension = 1 / s_dim
# 常量
M_J = 1.898e27 * M_dimension  # 木星质量
M_sun = 2.0e30 * M_dimension  # 太阳质量
G = 6.67e-11 * L_dimension / m_dim ** 3 / (M_dimension * T_dimension ** 2)  # m^3/(kg * s^2)
c = 3e8 * L_dimension / m_dim / T_dimension  # m/s
d = 2062650000 * 1.496e11 * L_dimension  # 源光度距离
R_earth = 1.496e11 * L_dimension  # 地球半径
P_earth = 31536000 * T_dimension  # 地球公转周期
year = 31536000 * T_dimension
# 常数
# i = math.pi / 3  # 康亚城设定的常数,源轨道倾角（随机分布）
psi_0 = 0  # rad 康亚城设定的常数,观测者初相位
phi_0 = 0  # 源轨道初相位

theta_s_h = math.acos(0.3)  # 源的坐标
phi_s_h = 5  # 源的坐标
theta_s = math.acos(0.3)  # 源的坐标
phi_s = 5  # 源的坐标
theta_l = math.acos(-0.2)  # 轨道坐标
phi_l = 4  # 轨道坐标

MM = math.pow(0.25 * par.M_b * par.M_b, 3 / 5) / math.pow(par.M_b, 0.2)  # 啁啾质量
f_dao = 96 / 5 * math.pow(G * MM / math.pow(c, 3), 5 / 3) * math.pow(math.pi, 8 / 3) * math.pow(par.f, 11 / 3)

# 功率谱密度

l_tian_qin = math.pow(3, 0.5) * 1e8 * L_dimension  # m
f_tian_qin = c / (2 * math.pi * l_tian_qin)
s_n_tianqin = 10 / 3 / l_tian_qin ** 2 * (
        (1e-24 * L_dimension ** 2 * T_dimension) + 4 * (1e-30 * L_dimension ** 2 / T_dimension ** 3) / (
        2 * math.pi * par.f / T_dimension) ** 4 * (1 + 1e-4 / par.f)) * (1 + 6 / 10 * (par.f / f_tian_qin) ** 2)

p_oms_lisa = 1.5e-11 ** 2 * (1 + (0.002 / T_dimension / par.f) ** 4) * L_dimension ** 2 * T_dimension
p_acc_lisa = 3e-15 ** 2 * (1 + (0.0004 / T_dimension / par.f) ** 2) * (
        1 + (par.f / 0.008 / T_dimension) ** 4) * L_dimension ** 2 / T_dimension ** 3
l_lisa = 2.5e9 * L_dimension
f_lisa = 0.01909 / T_dimension
s_n_lisa = 10 / 3 / l_lisa ** 2 * (
        p_oms_lisa + 2 * (1 + (math.cos(par.f / f_lisa)) ** 2) * p_acc_lisa / (2 * math.pi * par.f) ** 4) * (
                   1 + 6 / 10 * (par.f * f_lisa) ** 2)

p_oms_taiji = 8e-12 ** 2 * (1 + (0.002 / T_dimension / par.f) ** 4) * L_dimension ** 2 * T_dimension
p_acc_taiji = 3e-15 ** 2 * (1 + (0.0004 / T_dimension / par.f) ** 2) * (
        1 + (par.f / 0.008 / T_dimension) ** 4) * L_dimension ** 2 / T_dimension ** 3
l_taiji = 3e9 * L_dimension
f_taiji = c / (2 * math.pi * l_taiji)
s_n_taiji = 10 / 3 / l_taiji ** 2 * (
            p_oms_taiji + 2 * (1 + (math.cos(par.f / f_taiji)) ** 2) * p_acc_taiji / (2 * math.pi * par.f) ** 4) * (
            1 + 6 / 10 * (par.f * f_taiji) ** 2)

f_decigo = 7.36
s_n_decigo = (3 / 4 * 7.05e-48 * (1 + (par.f / f_decigo) ** 2) + (4.8E-51 * par.f ** (-4)) / (
                1 + (par.f / f_decigo) ** 2) +
            5.33E-52 * par.f ** (-4))

s_n_bbo = 2e-49 * par.f ** 2 + 4.58e-49 + 1.26e-51 * par.f ** (-4) * 3 / 4
