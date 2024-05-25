import math
def psi_s(t, p_theta_l, p_phi_l, p_theta_s, p_phi_s, p_phi_0):
    # return math.atan((0.5 * math.cos(p_theta_l) - math.sqrt(3) / 2 * math.sin(p_theta_l) * math.cos(
    #     2 * math.pi * t / P + p_phi_0 - p_phi_l)
    #                   - (math.cos(p_theta_l) * math.cos(p_theta_s) + math.sin(p_theta_l) * math.sin(
    #             p_theta_s) * math.cos(p_phi_l - p_phi_s))
    #                   * (math.sin(p_theta_s) * math.cos(p_phi_s) + math.sin(p_theta_s) * math.sin(
    #             p_phi_s) + math.cos(p_phi_s))) /
    #                  (0.5 * math.sin(p_theta_l) * math.sin(p_theta_s) * math.cos(p_phi_l - p_phi_s) -
    #                   math.sqrt(3) / 2 * math.cos(2 * math.pi * t / P + p_phi_0) * (
    #                           math.cos(p_theta_l) * math.sin(p_theta_s) * math.sin(p_phi_s) - math.cos(
    #                       p_theta_s) * math.sin(p_theta_l) * math.sin(p_phi_l)) -
    #                   math.sqrt(3) / 2 * math.cos(2 * math.pi * t / P + p_phi_0) * (
    #                           math.cos(p_theta_s) * math.sin(p_theta_l) * math.sin(p_phi_l) - math.cos(
    #                       p_theta_l) * math.sin(p_theta_s) * math.sin(p_phi_s))))
    return 0.399265977849603


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