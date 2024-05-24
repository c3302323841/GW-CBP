import math
import numpy as np
from scipy import integrate
from constant import phi_0, theta_s_h, phi_s_h, theta_s, phi_s, theta_l, phi_l
from parameter import f, T_obs, M_b
import constant as con


def fisher1(p_S_n_ni, p_P, p_M_p):
    # 定义函数以获得h1,h2-------------------------------------------------------------------------------------------
    def psi_s(p_theta_l, p_phi_l, p_theta_s, p_phi_s):
        return math.atan((math.cos(p_theta_s) * math.cos(p_phi_s - p_phi_l) * math.sin(p_theta_l) - math.cos(
            p_theta_l) * math.sin(p_theta_s)) / (math.sin(p_theta_l) * math.sin(p_phi_s - p_phi_l)))

    def i(p_theta_l, p_phi_l, p_theta_s, p_phi_s):
        return math.cos(p_theta_l) * math.cos(p_theta_s) + math.sin(p_theta_l) * math.sin(
            p_theta_s) * math.cos(p_phi_l - p_phi_s)

    K = math.pow(2 * math.pi * con.G / p_P, 1 / 3) * p_M_p / math.pow(p_M_p + M_b, 2 / 3) * math.sin(
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

    AA = 2 * math.pow(con.G * M_b
                      , 5 / 3) * math.pow(math.pi * f, 2 / 3) / (math.pow(con.c, 4) * con.d)

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
        return 2 * math.pi * t / con.P_earth + phi_0

    def f_obs(t):
        return (1 - K * math.cos(2 * math.pi * t / p_P + phi_0) / con.c) * (f + con.f_dao * t)

    def fiD(t):
        return 2 * math.pi * f_obs(t) * con.R_earth * math.sin(theta_s) * math.cos(
            phi_0 + 2 * math.pi * t / con.P_earth - phi_s) / con.c

    def psi_obs(t):
        return 2 * math.pi * (f + 0.5 * con.f_dao * t) * t - p_P * f * K / con.c * math.sin(
            fi_t(t)) - p_P * con.f_dao * t / con.c * K * math.sin(fi_t(t)) - p_P * p_P * con.f_dao * K / (
                2 * math.pi * con.c) * math.cos(
            fi_t(t))

    def h1(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.cos(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t))

    def h2(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.cos(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t))

    def find_delta_h1_theta_s():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s + delta, phi_s,
                                              psi_s(theta_l, phi_l, theta_s + delta, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s + delta, phi_s,
                                  psi_s(theta_l, phi_l, theta_s + delta, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s - delta, phi_s,
                                 psi_s(theta_l, phi_l, theta_s - delta, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s - delta, phi_s,
                                  psi_s(theta_l, phi_l, theta_s - delta, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        h1 = 1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        return h

    delta_h1_theta_s = find_delta_h1_theta_s()

    def find_delta_h1_phi_s():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s, phi_s + delta,
                                              psi_s(theta_l, phi_l, theta_s, phi_s + delta)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s, phi_s + delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s + delta)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s, phi_s - delta,
                                 psi_s(theta_l, phi_l, theta_s, phi_s - delta)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s, phi_s - delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s - delta)) + fiD(
                    x))) / (
                    2 * delta)

        h1 = 0.1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        return h

    delta_h1_phi_s = find_delta_h1_phi_s()

    def find_delta_h1_theta_l():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                              psi_s(theta_l + delta, phi_l, theta_s, phi_s,
                                                    )) * math.cos(
                psi_obs(x) + fi_1(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l + delta, phi_l, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                 psi_s(theta_l - delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l - delta, phi_l, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        h1 = 1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        return h

    delta_h1_theta_l = find_delta_h1_theta_l()

    def find_delta_h1_phi_l():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s, phi_s,
                                              psi_s(theta_l, phi_l + delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l, phi_l + delta, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s, phi_s,
                                 psi_s(theta_l, phi_l - delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l, phi_l - delta, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        h1 = 0.1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        return h

    delta_h1_phi_l = find_delta_h1_phi_l()

    def find_delta_h2_theta_s():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s + delta, phi_s,
                                              psi_s(theta_l, phi_l, theta_s + delta, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s + delta, phi_s,
                                  psi_s(theta_l, phi_l, theta_s + delta, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s - delta, phi_s,
                                 psi_s(theta_l, phi_l, theta_s - delta, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s - delta, phi_s,
                                  psi_s(theta_l, phi_l, theta_s - delta, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        h1 = 1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        return h

    delta_h2_theta_s = find_delta_h2_theta_s()

    def find_delta_h2_phi_s():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s, phi_s + delta,
                                              psi_s(theta_l, phi_l, theta_s, phi_s + delta)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s, phi_s + delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s + delta)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s, phi_s - delta,
                                 psi_s(theta_l, phi_l, theta_s, phi_s - delta)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s, phi_s - delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s - delta)) + fiD(
                    x))) / (
                    2 * delta)

        h1 = 0.1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        return h

    delta_h2_phi_s = find_delta_h2_phi_s()

    def find_delta_h2_theta_l():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                              psi_s(theta_l + delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l + delta, phi_l, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                 psi_s(theta_l - delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l - delta, phi_l, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        h1 = 1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        return h

    delta_h2_theta_l = find_delta_h2_theta_l()

    def find_delta_h2_phi_l():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s, phi_s,
                                              psi_s(theta_l, phi_l + delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l, phi_l + delta, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s, phi_s,
                                 psi_s(theta_l, phi_l - delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l, phi_l - delta, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        h1 = 0.1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        return h

    delta_h2_phi_l = find_delta_h2_phi_l()

    # 求偏导---------------------------------------------------------------------------------------------------------------
    def partial_h1_lnA(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.cos(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t))

    def partial_h1_psi_0(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t))

    def partial_h1_f(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s, )) * math.sin(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s, )) + fiD(t)) * (
                2 * math.pi * t - p_P / con.c * K * math.sin(fi_t(t)))

    def partial_h1_f_dao(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t)) * (
                math.pow(t, 2) * math.pi - p_P * t / con.c * K * math.sin(fi_t(t)) - math.pow(p_P,
                                                                                          2) * K / (
                        2 * math.pi * con.c) * math.cos(fi_t(t)))

    def partial_h1_theta_s(t):
        h = delta_h1_theta_s

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s + delta, phi_s,
                                              psi_s(theta_l, phi_l, theta_s + delta, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s + delta, phi_s,
                                  psi_s(theta_l, phi_l, theta_s + delta, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s - delta, phi_s,
                                 psi_s(theta_l, phi_l, theta_s - delta, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s - delta, phi_s,
                                  psi_s(theta_l, phi_l, theta_s - delta, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h1_phi_s(t):
        h = delta_h1_phi_s

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s, phi_s + delta,
                                              psi_s(theta_l, phi_l, theta_s, phi_s + delta)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s, phi_s + delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s + delta)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s, phi_s - delta,
                                 psi_s(theta_l, phi_l, theta_s, phi_s - delta)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s, phi_s - delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s - delta)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h1_theta_l(t):
        h = delta_h1_theta_l

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                              psi_s(theta_l + delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l + delta, phi_l, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                 psi_s(theta_l - delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l - delta, phi_l, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h1_phi_l(t):
        h = delta_h1_phi_l

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s, phi_s,
                                              psi_s(theta_l, phi_l + delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l, phi_l + delta, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s, phi_s,
                                 psi_s(theta_l, phi_l - delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_1(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l, phi_l - delta, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h1_K(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(
                t)) * (-1) * p_P * f / con.c * math.sin(
            fi_t(t))

    def partial_h1_P(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t)) * (
                (-1) * f * K / con.c * math.sin(fi_t(t)) + 2 * math.pi * f * t / (
                con.c * p_P) * K * math.cos(fi_t(t)))

    def partial_h1_phi_0(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_1(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(
                t)) * ((-1) * p_P * f * K / con.c * math.cos(fi_t(t)) - 2 * math.pi * f_obs(t) * con.R_earth * math.sin(
            theta_s) * math.cos(phi_0 + 2 * math.pi * t / con.P_earth - phi_s) / con.c)

    # h2的偏导——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    def partial_h2_lnA(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.cos(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t))

    def partial_h2_psi_0(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.cos(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t))

    def partial_h2_f(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t)) * (
                2 * math.pi * t - p_P / con.c * K * math.sin(fi_t(t)))

    def partial_h2_f_dao(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t)) * (
                math.pow(t, 2) * math.pi - p_P * t / con.c * K * math.sin(fi_t(t)) - math.pow(p_P,
                                                                                          2) * K / (
                        2 * math.pi * con.c) * math.cos(fi_t(t)))

    def partial_h2_theta_s(t):
        h = delta_h2_theta_s

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s + delta, phi_s,
                                              psi_s(theta_l, phi_l, theta_s + delta, phi_s)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l, phi_l, theta_s + delta, phi_s), theta_s + delta, phi_s,
                                  psi_s(theta_l, phi_l, theta_s + delta, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s - delta, phi_s,
                                 psi_s(theta_l, phi_l, theta_s - delta, phi_s)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l, phi_l, theta_s - delta, phi_s), theta_s - delta, phi_s,
                                  psi_s(theta_l, phi_l, theta_s - delta, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h2_phi_s(t):
        h = delta_h2_phi_s

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s, phi_s + delta,
                                              psi_s(theta_l, phi_l, theta_s, phi_s + delta)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l, phi_l, theta_s, phi_s + delta), theta_s, phi_s + delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s + delta)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s, phi_s - delta,
                                 psi_s(theta_l, phi_l, theta_s, phi_s - delta)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l, phi_l, theta_s, phi_s - delta), theta_s, phi_s - delta,
                                  psi_s(theta_l, phi_l, theta_s, phi_s - delta)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h2_theta_l(t):
        h = delta_h2_theta_l

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                              psi_s(theta_l + delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l + delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l + delta, phi_l, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                 psi_s(theta_l - delta, phi_l, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l - delta, phi_l, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l - delta, phi_l, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h2_phi_l(t):
        h = delta_h2_phi_l

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s, phi_s,
                                              psi_s(theta_l, phi_l + delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l, phi_l + delta, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l, phi_l + delta, theta_s, phi_s)) + fiD(
                    x)) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s, phi_s,
                                 psi_s(theta_l, phi_l - delta, theta_s, phi_s)) * math.cos(
                psi_obs(x) + fi_2(i(theta_l, phi_l - delta, theta_s, phi_s), theta_s, phi_s,
                                  psi_s(theta_l, phi_l - delta, theta_s, phi_s)) + fiD(
                    x))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h2_K(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(
                t)) * (-1) * p_P * f / con.c * math.sin(
            fi_t(t))

    def partial_h2_P(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(t)) * (
                (-1) * f * K / con.c * math.sin(fi_t(t)) + 2 * math.pi * f * t / (
                con.c * p_P) * K * math.cos(
            fi_t(t)))

    def partial_h2_phi_0(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                                         psi_s(theta_l, phi_l, theta_s, phi_s)) * math.sin(
            psi_obs(t) + fi_2(i(theta_l, phi_l, theta_s, phi_s), theta_s, phi_s,
                              psi_s(theta_l, phi_l, theta_s, phi_s)) + fiD(
                t)) * ((-1) * p_P * f * K / con.c * math.cos(fi_t(t)) - 2 * math.pi * f_obs(t) * con.R_earth * math.sin(
            theta_s) * math.cos(
            phi_0 + 2 * math.pi * t / con.P_earth - phi_s) / con.c)

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


def fisher2(p_S_n_ni, p_P, p_M_p):
    # 定义函数以获得h1,h2-------------------------------------------------------------------------------------------------
    def psi_s(t, p_theta_s_h, p_phi_s_h, p_theta_l, p_phi_l):
        return math.atan((0.5 * math.cos(p_theta_l) - math.sqrt(3) / 2 * math.sin(p_theta_l) * math.cos(
            2 * math.pi * t / p_P + phi_0 - p_phi_l)
                          - (math.cos(p_theta_l) * math.cos(p_theta_s_h) - math.sin(p_theta_l) * math.sin(
                    p_theta_s_h) * math.cos(p_phi_l - p_phi_s_h))) /
                         (0.5 * math.sin(p_theta_l) * math.sin(p_theta_s_h) * math.sin(p_phi_l - p_phi_s_h) -
                          math.sqrt(3) / 2 * math.cos(2 * math.pi * t / p_P + phi_0) * (
                                  math.cos(p_theta_l) * math.sin(p_theta_s_h) * math.sin(p_phi_s_h) - math.cos(
                              p_theta_s_h) * math.sin(p_theta_l) * math.sin(p_phi_l)) -
                          math.sqrt(3) / 2 * math.cos(2 * math.pi * t / p_P + phi_0) * (
                                  math.cos(p_theta_s_h) * math.sin(p_theta_l) * math.cos(p_phi_l) - math.cos(
                              p_theta_l) * math.sin(p_theta_s_h) * math.sin(p_phi_s_h))))

    def i(p_theta_l, p_phi_l, p_theta_s, p_phi_s):
        return math.cos(p_theta_l) * math.cos(p_theta_s) + math.sin(p_theta_l) * math.sin(
            p_theta_s) * math.cos(p_phi_l - p_phi_s)

    def K(p_i):
        return math.pow(2 * math.pi * con.G / p_P, 1 / 3) * p_M_p / math.pow(p_M_p + M_b
                                                                             , 2 / 3) * math.sin(
            p_i)

    def theta_s(t, p_theta_s_h, p_phi_s_h):
        return math.acos(
            0.5 * math.cos(p_theta_s_h) - 0.5 * pow(3, 0.5) * math.sin(p_theta_s_h) * math.cos(
                fi_t(t) - p_phi_s_h))

    def phi_s(t, p_theta_s_h, p_phi_s_h):
        return 2 * math.pi * t / con.P_earth - math.atan(
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

    AA = 2 * math.pow(con.G * con.MM, 5 / 3) * math.pow(math.pi * f, 2 / 3) / (math.pow(con.c, 4) * con.d)

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
        return 2 * math.pi * t / con.P_earth + phi_0

    def f_obs(t, p_i):
        return (1 - K(p_i) * math.cos(2 * math.pi * t / p_P + phi_0) / con.c) * (f + con.f_dao * t)

    def fiD(t, p_i):
        return 2 * math.pi * f_obs(t, p_i) * con.R_earth * math.sin(theta_s_h) * math.cos(
            phi_0 + 2 * math.pi * t / con.P_earth - phi_s_h) / con.c

    def psi_obs(t, p_i):
        return 2 * math.pi * (f + 0.5 * con.f_dao * t) * t - p_P * f * K(p_i) / con.c * math.sin(
            fi_t(t)) - p_P * con.f_dao * t / con.c * K(p_i) * math.sin(fi_t(t)) - p_P * p_P * con.f_dao * K(p_i) / (
                2 * math.pi * con.c) * math.cos(
            fi_t(t))

    def h1(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.cos(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_1(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l,
                                                                                              theta_s_h,
                                                                                              phi_s_h)))

    def h2(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.cos(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_2(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l,
                                                                                              theta_s_h,
                                                                                              phi_s_h)))

    def find_delta_h1_theta_s():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h + delta, phi_s_h),
                                              theta_s(x, theta_s_h + delta, phi_s_h),
                                              phi_s(x, theta_s_h + delta, phi_s_h),
                                              psi_s(x, theta_s_h + delta, phi_s_h, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h + delta, phi_s_h)) + fi_1(
                    i(theta_l, phi_l, theta_s_h + delta, phi_s_h), theta_s(x, theta_s_h + delta, phi_s_h),
                    phi_s(x, theta_s_h + delta, phi_s_h),
                    psi_s(x, theta_s_h + delta, phi_s_h, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h + delta, phi_s_h))) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h - delta, phi_s_h),
                                 theta_s(x, theta_s_h - delta, phi_s_h), phi_s(x, theta_s_h - delta, phi_s_h),
                                 psi_s(x, theta_s_h - delta, phi_s_h, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h - delta, phi_s_h)) + fi_1(
                    i(theta_l, phi_l, theta_s_h - delta, phi_s_h), theta_s(x, theta_s_h - delta, phi_s_h),
                    phi_s(x, theta_s_h - delta, phi_s_h),
                    psi_s(x, theta_s_h - delta, phi_s_h, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h - delta, phi_s_h)))) / (
                    2 * delta)

        h1 = 1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        return h

    delta_h1_theta_s = find_delta_h1_theta_s()

    def find_delta_h1_phi_s():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h + delta),
                                              theta_s(x, theta_s_h, phi_s_h + delta),
                                              phi_s(x, theta_s_h, phi_s_h + delta),
                                              psi_s(x, theta_s_h, phi_s_h + delta, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h, phi_s_h + delta)) + fi_1(
                    i(theta_l, phi_l, theta_s_h, phi_s_h + delta), theta_s(x, theta_s_h, phi_s_h + delta),
                    phi_s(x, theta_s_h, phi_s_h + delta),
                    psi_s(x, theta_s_h, phi_s_h + delta, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h, phi_s_h + delta))) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h - delta),
                                 theta_s(x, theta_s_h, phi_s_h - delta), phi_s(x, theta_s_h, phi_s_h - delta),
                                 psi_s(x, theta_s_h, phi_s_h - delta, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h, phi_s_h - delta)) + fi_1(
                    i(theta_l, phi_l, theta_s_h, phi_s_h - delta), theta_s(x, theta_s_h, phi_s_h - delta),
                    phi_s(x, theta_s_h, phi_s_h - delta),
                    psi_s(x, theta_s_h, phi_s_h - delta, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h, phi_s_h - delta)))) / (
                    2 * delta)

        h1 = 1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1

        return h

    delta_h1_phi_s = find_delta_h1_phi_s()

    def find_delta_h1_theta_l():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l + delta, phi_l, theta_s_h, phi_s_h),
                                              theta_s(x, theta_s_h, phi_s_h), phi_s(x, theta_s_h, phi_s_h),
                                              psi_s(x, theta_s_h, phi_s_h, theta_l + delta, phi_l)) * math.cos(
                psi_obs(x, i(theta_l + delta, phi_l, theta_s_h, phi_s_h)) + fi_1(
                    i(theta_l + delta, phi_l, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l + delta, phi_l)) + fiD(
                    x, i(theta_l + delta, phi_l, theta_s_h, phi_s_h))) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l - delta, phi_l, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                                 phi_s(x, theta_s_h, phi_s_h),
                                 psi_s(x, theta_s_h, phi_s_h, theta_l - delta, phi_l)) * math.cos(
                psi_obs(x, i(theta_l - delta, phi_l, theta_s_h, phi_s_h)) + fi_1(
                    i(theta_l - delta, phi_l, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l - delta, phi_l)) + fiD(
                    x, i(theta_l - delta, phi_l, theta_s_h, phi_s_h)))) / (
                    2 * delta)

        h1 = 1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        return h

    delta_h1_theta_l = find_delta_h1_theta_l()

    def find_delta_h1_phi_l():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l + delta, theta_s_h, phi_s_h),
                                              theta_s(x, theta_s_h, phi_s_h), phi_s(x, theta_s_h, phi_s_h),
                                              psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l + delta)) * math.cos(
                psi_obs(x, i(theta_l, phi_l + delta, theta_s_h, phi_s_h)) + fi_1(
                    i(theta_l, phi_l + delta, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l + delta)) + fiD(
                    x, i(theta_l, phi_l + delta, theta_s_h, phi_s_h))) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l - delta, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                                 phi_s(x, theta_s_h, phi_s_h),
                                 psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l - delta)) * math.cos(
                psi_obs(x, i(theta_l, phi_l - delta, theta_s_h, phi_s_h)) + fi_1(
                    i(theta_l, phi_l - delta, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l - delta)) + fiD(
                    x, i(theta_l, phi_l - delta, theta_s_h, phi_s_h)))) / (
                    2 * delta)

        h1 = 1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        # print('h4=', h)
        return h

    delta_h1_phi_l = find_delta_h1_phi_l()

    # delta_h1_phi_l = 1e-4

    def find_delta_h2_theta_s():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h + delta, phi_s_h),
                                              theta_s(x, theta_s_h + delta, phi_s_h),
                                              phi_s(x, theta_s_h + delta, phi_s_h),
                                              psi_s(x, theta_s_h + delta, phi_s_h, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h + delta, phi_s_h)) + fi_2(
                    i(theta_l, phi_l, theta_s_h + delta, phi_s_h), theta_s(x, theta_s_h + delta, phi_s_h),
                    phi_s(x, theta_s_h + delta, phi_s_h),
                    psi_s(x, theta_s_h + delta, phi_s_h, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h + delta, phi_s_h))) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h - delta, phi_s_h),
                                 theta_s(x, theta_s_h - delta, phi_s_h), phi_s(x, theta_s_h - delta, phi_s_h),
                                 psi_s(x, theta_s_h - delta, phi_s_h, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h - delta, phi_s_h)) + fi_2(
                    i(theta_l, phi_l, theta_s_h - delta, phi_s_h), theta_s(x, theta_s_h - delta, phi_s_h),
                    phi_s(x, theta_s_h - delta, phi_s_h),
                    psi_s(x, theta_s_h - delta, phi_s_h, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h - delta, phi_s_h)))) / (
                    2 * delta)

        h1 = 1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        # print('h5=', h)
        return h

    delta_h2_theta_s = find_delta_h2_theta_s()

    # delta_h2_theta_s = 1e-4

    def find_delta_h2_phi_s():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h, phi_s_h + delta),
                                              theta_s(x, theta_s_h, phi_s_h + delta),
                                              phi_s(x, theta_s_h, phi_s_h + delta),
                                              psi_s(x, theta_s_h, phi_s_h + delta, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h, phi_s_h + delta)) + fi_2(
                    i(theta_l, phi_l, theta_s_h, phi_s_h + delta), theta_s(x, theta_s_h, phi_s_h + delta),
                    phi_s(x, theta_s_h, phi_s_h + delta),
                    psi_s(x, theta_s_h, phi_s_h + delta, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h, phi_s_h + delta))) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h, phi_s_h - delta),
                                 theta_s(x, theta_s_h, phi_s_h - delta), phi_s(x, theta_s_h, phi_s_h - delta),
                                 psi_s(x, theta_s_h, phi_s_h - delta, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h, phi_s_h - delta)) + fi_2(
                    i(theta_l, phi_l, theta_s_h, phi_s_h - delta), theta_s(x, theta_s_h, phi_s_h - delta),
                    phi_s(x, theta_s_h, phi_s_h - delta),
                    psi_s(x, theta_s_h, phi_s_h - delta, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h, phi_s_h - delta)))) / (
                    2 * delta)

        h1 = 1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        # print('h6=', h)
        return h

    delta_h2_phi_s = find_delta_h2_phi_s()

    # delta_h2_phi_s = 1e-5

    def find_delta_h2_theta_l():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l + delta, phi_l, theta_s_h, phi_s_h),
                                              theta_s(x, theta_s_h, phi_s_h), phi_s(x, theta_s_h, phi_s_h),
                                              psi_s(x, theta_s_h, phi_s_h, theta_l + delta, phi_l)) * math.cos(
                psi_obs(x, i(theta_l + delta, phi_l, theta_s_h, phi_s_h)) + fi_2(
                    i(theta_l + delta, phi_l, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l + delta, phi_l)) + fiD(
                    x, i(theta_l + delta, phi_l, theta_s_h, phi_s_h))) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l - delta, phi_l, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                                 phi_s(x, theta_s_h, phi_s_h),
                                 psi_s(x, theta_s_h, phi_s_h, theta_l - delta, phi_l)) * math.cos(
                psi_obs(x, i(theta_l - delta, phi_l, theta_s_h, phi_s_h)) + fi_2(
                    i(theta_l - delta, phi_l, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l - delta, phi_l)) + fiD(
                    x, i(theta_l - delta, phi_l, theta_s_h, phi_s_h)))) / (
                    2 * delta)

        h1 = 1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        # print('h7=', h)
        return h

    delta_h2_theta_l = find_delta_h2_theta_l()

    # delta_h2_theta_l = 1e-4

    def find_delta_h2_phi_l():
        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l + delta, theta_s_h, phi_s_h),
                                              theta_s(x, theta_s_h, phi_s_h), phi_s(x, theta_s_h, phi_s_h),
                                              psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l + delta)) * math.cos(
                psi_obs(x, i(theta_l, phi_l + delta, theta_s_h, phi_s_h)) + fi_2(
                    i(theta_l, phi_l + delta, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l + delta)) + fiD(
                    x, i(theta_l, phi_l + delta, theta_s_h, phi_s_h))) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l - delta, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                                 phi_s(x, theta_s_h, phi_s_h),
                                 psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l - delta)) * math.cos(
                psi_obs(x, i(theta_l, phi_l - delta, theta_s_h, phi_s_h)) + fi_2(
                    i(theta_l, phi_l - delta, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l - delta)) + fiD(
                    x, i(theta_l, phi_l - delta, theta_s_h, phi_s_h)))) / (
                    2 * delta)

        h1 = 1
        v3 = 1
        while v3 > 0.00001:
            y = lambda x: par_1(x, h1) * par_1(x, h1)
            v1 = integrate.quad(y, 0, T_obs
)[0] * 2 * p_S_n_ni
            h2 = 0.1 * h1
            y1 = lambda x: par_1(x, h2) * par_1(x, h2)
            v2 = integrate.quad(y1, 0, T_obs
)[0] * 2 * p_S_n_ni
            if v1 - v2 == 0:
                break
            v3 = abs((v1 - v2) / v1)
            h1 = h2
            # print(h1)
        h = h1
        # print('h8=', h)
        return h

    delta_h2_phi_l = find_delta_h2_phi_l()

    # delta_h2_phi_l = 1e-4
    # 求偏导---------------------------------------------------------------------------------------------------------------
    def partial_h1_lnA(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.cos(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_1(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l,
                                                                                              theta_s_h,
                                                                                              phi_s_h)))

    def partial_h1_psi_0(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.sin(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_1(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l,
                                                                                              theta_s_h,
                                                                                              phi_s_h)))

    def partial_h1_f(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.sin(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_1(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l,
                                                                                              theta_s_h,
                                                                                              phi_s_h))) * (
                2 * math.pi * t - p_P / con.c * K(i(theta_l, phi_l, theta_s_h, phi_s_h)) * math.sin(fi_t(t)))

    def partial_h1_f_dao(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.sin(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_1(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l,
                                                                                              theta_s_h,
                                                                                              phi_s_h))) * (
                math.pow(t, 2) * math.pi - p_P * t / con.c * K(i(theta_l, phi_l, theta_s_h, phi_s_h)) * math.sin(
            fi_t(t)) - math.pow(p_P,
                                2) * K(i(theta_l, phi_l, theta_s_h, phi_s_h)) / (
                        2 * math.pi * con.c) * math.cos(fi_t(t)))

    def partial_h1_theta_s(t):
        h = delta_h1_theta_s

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h + delta, phi_s_h),
                                              theta_s(x, theta_s_h + delta, phi_s_h),
                                              phi_s(x, theta_s_h + delta, phi_s_h),
                                              psi_s(x, theta_s_h + delta, phi_s_h, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h + delta, phi_s_h)) + fi_1(
                    i(theta_l, phi_l, theta_s_h + delta, phi_s_h), theta_s(x, theta_s_h + delta, phi_s_h),
                    phi_s(x, theta_s_h + delta, phi_s_h),
                    psi_s(x, theta_s_h + delta, phi_s_h, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h + delta, phi_s_h))) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h - delta, phi_s_h),
                                 theta_s(x, theta_s_h - delta, phi_s_h), phi_s(x, theta_s_h - delta, phi_s_h),
                                 psi_s(x, theta_s_h - delta, phi_s_h, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h - delta, phi_s_h)) + fi_1(
                    i(theta_l, phi_l, theta_s_h - delta, phi_s_h), theta_s(x, theta_s_h - delta, phi_s_h),
                    phi_s(x, theta_s_h - delta, phi_s_h),
                    psi_s(x, theta_s_h - delta, phi_s_h, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h - delta, phi_s_h)))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h1_phi_s(t):
        h = delta_h1_phi_s

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h + delta),
                                              theta_s(x, theta_s_h, phi_s_h + delta),
                                              phi_s(x, theta_s_h, phi_s_h + delta),
                                              psi_s(x, theta_s_h, phi_s_h + delta, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h, phi_s_h + delta)) + fi_1(
                    i(theta_l, phi_l, theta_s_h, phi_s_h + delta), theta_s(x, theta_s_h, phi_s_h + delta),
                    phi_s(x, theta_s_h, phi_s_h + delta),
                    psi_s(x, theta_s_h, phi_s_h + delta, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h, phi_s_h + delta))) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h - delta),
                                 theta_s(x, theta_s_h, phi_s_h - delta), phi_s(x, theta_s_h, phi_s_h - delta),
                                 psi_s(x, theta_s_h, phi_s_h - delta, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h, phi_s_h - delta)) + fi_1(
                    i(theta_l, phi_l, theta_s_h, phi_s_h - delta), theta_s(x, theta_s_h, phi_s_h - delta),
                    phi_s(x, theta_s_h, phi_s_h - delta),
                    psi_s(x, theta_s_h, phi_s_h - delta, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h, phi_s_h - delta)))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h1_theta_l(t):
        h = delta_h1_theta_l

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l + delta, phi_l, theta_s_h, phi_s_h),
                                              theta_s(x, theta_s_h, phi_s_h), phi_s(x, theta_s_h, phi_s_h),
                                              psi_s(x, theta_s_h, phi_s_h, theta_l + delta, phi_l)) * math.cos(
                psi_obs(x, i(theta_l + delta, phi_l, theta_s_h, phi_s_h)) + fi_1(
                    i(theta_l + delta, phi_l, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l + delta, phi_l)) + fiD(
                    x, i(theta_l + delta, phi_l, theta_s_h, phi_s_h))) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l - delta, phi_l, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                                 phi_s(x, theta_s_h, phi_s_h),
                                 psi_s(x, theta_s_h, phi_s_h, theta_l - delta, phi_l)) * math.cos(
                psi_obs(x, i(theta_l - delta, phi_l, theta_s_h, phi_s_h)) + fi_1(
                    i(theta_l - delta, phi_l, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l - delta, phi_l)) + fiD(
                    x, i(theta_l - delta, phi_l, theta_s_h, phi_s_h)))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h1_phi_l(t):
        h = delta_h1_phi_l

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l + delta, theta_s_h, phi_s_h),
                                              theta_s(x, theta_s_h, phi_s_h), phi_s(x, theta_s_h, phi_s_h),
                                              psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l + delta)) * math.cos(
                psi_obs(x, i(theta_l, phi_l + delta, theta_s_h, phi_s_h)) + fi_1(
                    i(theta_l, phi_l + delta, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l + delta)) + fiD(
                    x, i(theta_l, phi_l + delta, theta_s_h, phi_s_h))) - math.pow(
                3, 0.5) / 2 * A1(i(theta_l, phi_l - delta, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                                 phi_s(x, theta_s_h, phi_s_h),
                                 psi_s(x, theta_s_h - delta, phi_s_h, theta_l, phi_l - delta)) * math.cos(
                psi_obs(x, i(theta_l, phi_l - delta, theta_s_h, phi_s_h)) + fi_1(
                    i(theta_l, phi_l - delta, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h - delta, phi_s_h, theta_l, phi_l - delta)) + fiD(
                    x, i(theta_l, phi_l - delta, theta_s_h, phi_s_h)))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h1_K(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.sin(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_1(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(
                t, i(theta_l, phi_l, theta_s_h, phi_s_h))) * (-1) * p_P * f / con.c * math.sin(
            fi_t(t))

    def partial_h1_P(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.sin(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_1(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l,
                                                                                              theta_s_h,
                                                                                              phi_s_h))) * (
                (-1) * f * K(i(theta_l, phi_l, theta_s_h, phi_s_h)) / con.c * math.sin(
            fi_t(t)) + 2 * math.pi * f * t / (
                        con.c * p_P) * K(i(theta_l, phi_l, theta_s_h, phi_s_h)) * math.cos(fi_t(t)))

    def partial_h1_phi_0(t):
        return math.pow(3, 0.5) / 2 * A1(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.sin(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_1(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(
                t, i(theta_l, phi_l, theta_s_h, phi_s_h))) * (
                (-1) * p_P * f * K(i(theta_l, phi_l, theta_s_h, phi_s_h)) / con.c * math.cos(
            fi_t(t)) - 2 * math.pi * f_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) * con.R_earth * math.sin(
            theta_s(t, theta_s_h, phi_s_h)) * math.cos(
            phi_0 + 2 * math.pi * t / con.P_earth - phi_s(t, theta_s_h, phi_s_h)) / con.c)

    # h2的偏导——————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    def partial_h2_lnA(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.cos(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_2(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l,
                                                                                              theta_s_h,
                                                                                              phi_s_h)))

    def partial_h2_psi_0(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.cos(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_2(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l,
                                                                                              theta_s_h,
                                                                                              phi_s_h)))

    def partial_h2_f(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.sin(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_2(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l,
                                                                                              theta_s_h,
                                                                                              phi_s_h))) * (
                2 * math.pi * t - p_P / con.c * K(i(theta_l, phi_l, theta_s_h, phi_s_h)) * math.sin(fi_t(t)))

    def partial_h2_f_dao(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.sin(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_2(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l,
                                                                                              theta_s_h,
                                                                                              phi_s_h))) * (
                math.pow(t, 2) * math.pi - p_P * t / con.c * K(i(theta_l, phi_l, theta_s_h, phi_s_h)) * math.sin(
            fi_t(t)) - math.pow(p_P,
                                2) * K(i(theta_l, phi_l, theta_s_h, phi_s_h)) / (
                        2 * math.pi * con.c) * math.cos(fi_t(t)))

    def partial_h2_theta_s(t):
        h = delta_h2_theta_s

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h + delta, phi_s_h),
                                              theta_s(x, theta_s_h + delta, phi_s_h),
                                              phi_s(x, theta_s_h + delta, phi_s_h),
                                              psi_s(x, theta_s_h + delta, phi_s_h, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h + delta, phi_s_h)) + fi_2(
                    i(theta_l, phi_l, theta_s_h + delta, phi_s_h), theta_s(x, theta_s_h + delta, phi_s_h),
                    phi_s(x, theta_s_h + delta, phi_s_h),
                    psi_s(x, theta_s_h + delta, phi_s_h, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h + delta, phi_s_h))) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h - delta, phi_s_h),
                                 theta_s(x, theta_s_h - delta, phi_s_h), phi_s(x, theta_s_h - delta, phi_s_h),
                                 psi_s(x, theta_s_h - delta, phi_s_h, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h - delta, phi_s_h)) + fi_2(
                    i(theta_l, phi_l, theta_s_h - delta, phi_s_h), theta_s(x, theta_s_h - delta, phi_s_h),
                    phi_s(x, theta_s_h - delta, phi_s_h),
                    psi_s(x, theta_s_h - delta, phi_s_h, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h - delta, phi_s_h)))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h2_phi_s(t):
        h = delta_h2_phi_s

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h, phi_s_h + delta),
                                              theta_s(x, theta_s_h, phi_s_h + delta),
                                              phi_s(x, theta_s_h, phi_s_h + delta),
                                              psi_s(x, theta_s_h, phi_s_h + delta, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h, phi_s_h + delta)) + fi_2(
                    i(theta_l, phi_l, theta_s_h, phi_s_h + delta), theta_s(x, theta_s_h, phi_s_h + delta),
                    phi_s(x, theta_s_h, phi_s_h + delta),
                    psi_s(x, theta_s_h, phi_s_h + delta, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h, phi_s_h + delta))) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h, phi_s_h - delta),
                                 theta_s(x, theta_s_h, phi_s_h - delta), phi_s(x, theta_s_h, phi_s_h - delta),
                                 psi_s(x, theta_s_h, phi_s_h - delta, theta_l, phi_l)) * math.cos(
                psi_obs(x, i(theta_l, phi_l, theta_s_h, phi_s_h - delta)) + fi_2(
                    i(theta_l, phi_l, theta_s_h, phi_s_h - delta), theta_s(x, theta_s_h, phi_s_h - delta),
                    phi_s(x, theta_s_h, phi_s_h - delta),
                    psi_s(x, theta_s_h, phi_s_h - delta, theta_l, phi_l)) + fiD(
                    x, i(theta_l, phi_l, theta_s_h, phi_s_h - delta)))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h2_theta_l(t):
        h = delta_h2_theta_l

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l + delta, phi_l, theta_s_h, phi_s_h),
                                              theta_s(x, theta_s_h, phi_s_h), phi_s(x, theta_s_h, phi_s_h),
                                              psi_s(x, theta_s_h, phi_s_h, theta_l + delta, phi_l)) * math.cos(
                psi_obs(x, i(theta_l + delta, phi_l, theta_s_h, phi_s_h)) + fi_2(
                    i(theta_l + delta, phi_l, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l + delta, phi_l)) + fiD(
                    x, i(theta_l + delta, phi_l, theta_s_h, phi_s_h))) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l - delta, phi_l, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                                 phi_s(x, theta_s_h, phi_s_h),
                                 psi_s(x, theta_s_h, phi_s_h, theta_l - delta, phi_l)) * math.cos(
                psi_obs(x, i(theta_l - delta, phi_l, theta_s_h, phi_s_h)) + fi_2(
                    i(theta_l - delta, phi_l, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l - delta, phi_l)) + fiD(
                    x, i(theta_l - delta, phi_l, theta_s_h, phi_s_h)))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h2_phi_l(t):
        h = delta_h2_phi_l

        def par_1(x, delta):
            return (math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l + delta, theta_s_h, phi_s_h),
                                              theta_s(x, theta_s_h, phi_s_h), phi_s(x, theta_s_h, phi_s_h),
                                              psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l + delta)) * math.cos(
                psi_obs(x, i(theta_l, phi_l + delta, theta_s_h, phi_s_h)) + fi_2(
                    i(theta_l, phi_l + delta, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l + delta)) + fiD(
                    x, i(theta_l, phi_l + delta, theta_s_h, phi_s_h))) - math.pow(
                3, 0.5) / 2 * A2(i(theta_l, phi_l - delta, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                                 phi_s(x, theta_s_h, phi_s_h),
                                 psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l - delta)) * math.cos(
                psi_obs(x, i(theta_l, phi_l - delta, theta_s_h, phi_s_h)) + fi_2(
                    i(theta_l, phi_l - delta, theta_s_h, phi_s_h), theta_s(x, theta_s_h, phi_s_h),
                    phi_s(x, theta_s_h, phi_s_h),
                    psi_s(x, theta_s_h, phi_s_h, theta_l, phi_l - delta)) + fiD(
                    x, i(theta_l, phi_l - delta, theta_s_h, phi_s_h)))) / (
                    2 * delta)

        return par_1(t, h)

    def partial_h2_K(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.sin(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_2(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(
                t, i(theta_l, phi_l, theta_s_h, phi_s_h))) * (-1) * p_P * f / con.c * math.sin(
            fi_t(t))

    def partial_h2_P(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.sin(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_2(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(t, i(theta_l, phi_l,
                                                                                              theta_s_h,
                                                                                              phi_s_h))) * (
                (-1) * f * K(i(theta_l, phi_l, theta_s_h, phi_s_h)) / con.c * math.sin(
            fi_t(t)) + 2 * math.pi * f * t / (
                con.c * p_P) * K(i(theta_l, phi_l, theta_s_h, phi_s_h)) * math.cos(
            fi_t(t)))

    def partial_h2_phi_0(t):
        return math.pow(3, 0.5) / 2 * A2(i(theta_l, phi_l, theta_s_h, phi_s_h), theta_s(t, theta_s_h, phi_s_h),
                                         phi_s(t, theta_s_h, phi_s_h),
                                         psi_s(t, theta_s_h, phi_s_h, theta_l, phi_l)) * math.sin(
            psi_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) + fi_2(i(theta_l, phi_l, theta_s_h, phi_s_h),
                                                                     theta_s(t, theta_s_h, phi_s_h),
                                                                     phi_s(t, theta_s_h, phi_s_h),
                                                                     psi_s(t, theta_s_h, phi_s_h, theta_l,
                                                                           phi_l)) + fiD(
                t, i(theta_l, phi_l, theta_s_h, phi_s_h))) * (
                (-1) * p_P * f * K(i(theta_l, phi_l, theta_s_h, phi_s_h)) / con.c * math.cos(
            fi_t(t)) - 2 * math.pi * f_obs(t, i(theta_l, phi_l, theta_s_h, phi_s_h)) * con.R_earth * math.sin(
            theta_s(t, theta_s_h, phi_s_h)) * math.cos(
            phi_0 + 2 * math.pi * t / con.P_earth - phi_s(t, theta_s_h, phi_s_h)) / con.c)

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
    # 求fisher矩阵
    for k in range(len(partial_h1)):
        for n in range(len(partial_h2)):
            y = lambda t: partial_h1[k](t) * partial_h1[n](t) + partial_h2[k](t) * partial_h2[n](t)
            v = integrate.quad(y, 0, T_obs)
            FF[k][n] = v[0] * 2 * p_S_n_ni

    return FF
