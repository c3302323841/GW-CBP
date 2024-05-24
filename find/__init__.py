import math
import numpy as np
from scipy import linalg
import fisher
import parameter as par
import constant as con


def find_m_p(na, nb, nc):
    detector = ''
    P_x = []
    R_x = []
    M_p_y = []
    for a in range(na):  # 循环周期
        P = math.pow(10, a * 0.2 - 1.6) * con.year
        # 误差图坐标列表
        b_x = []
        Re_y = []
        Re_z = []
        for b in range(nb):  # 寻找最小质量
            M_p = math.pow(10, (b / 100 * 3 - 1)) * con.M_J
            KK = math.pow(2 * math.pi * con.G / P, 1 / 3) * M_p / math.pow(M_p + par.M_b, 2 / 3) * math.sin(math.pi / 3)
            # R = math.pow(G * P ** 2 * (M_b + M_p) / (4 * math.pi ** 2), 1 / 3)

            fisher_tianqin = fisher.fisher1(1 / con.s_n_tianqin, P, M_p)
            fisher_taiji = fisher.fisher2(1 / con.s_n_taiji, P, M_p)
            fisher_lisa = fisher.fisher2(1 / con.s_n_lisa, P, M_p)

            # detector: tianqin is 1, taiji ji 2, lisa is 3, tianqin + taiji is 4, tianqin + lisa is 5
            # lisa +taiji is 6, three is 7
            if nc == 1:
                fisher_total = fisher_tianqin
                detector = 'Tianqin'
            elif nc == 2:
                fisher_total = fisher_taiji
                detector = 'Taiji'
            elif nc == 3:
                fisher_total = fisher_lisa
                detector = 'LISA'
            elif nc == 4:
                fisher_total = fisher_tianqin + fisher_taiji
                detector = 'Tianqin&Taiji'
            elif nc == 5:
                fisher_total = fisher_tianqin + fisher_lisa
                detector = 'Tianqin%LISA'
            elif nc == 6:
                fisher_total = fisher_lisa + fisher_taiji
                detector = 'LISA&Taiji'
            else:
                fisher_total = fisher_tianqin + fisher_taiji + fisher_lisa
                detector = 'Tianqin&Taiji&LISA'

            def var():
                covariance = linalg.inv(fisher_total)
                # 求逆出现负对角元，剔除之
                if covariance[8][8] < 0:
                    var_k = KK
                else:
                    var_k = math.pow(covariance[8][8], 1 / 2)  # variance 方差
                return var_k

            Re = var() / KK  # relative error 相对误差
            # 画误差图
            b_x.append(b)
            Re_y.append(math.log10(Re))
            Re_z.append(Re)

            # 误差符合条件时加入质量周期图，并开始下一个周期计算
            if b > 0 and Re < 0.3 and Re_z[b] / Re_z[b - 1] > 0.5:
                print(a, b)
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
                MP = math.log10(math.pow(10, y_predict / 100 * 3 - 1))
                P_x.append(math.log10(P / con.year))
                # R_x.append(math.log10(R / AU))
                M_p_y.append(MP)
                break
    return P_x, M_p_y, detector
