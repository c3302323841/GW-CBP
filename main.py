import matplotlib
import matplotlib.pyplot as plt
import find
import parameter as par
# import pandas as pd
# 切换为图形界面显示的终端TkAgg
matplotlib.use('TkAgg')

P = find.find_m_p(par.na, par.nb, par.nc)[0]
M_p = find.find_m_p(par.na, par.nb, par.nc)[1]
detector = find.find_m_p(par.na, par.nb, par.nc)[2]
# 画质量周期图
plt.plot(P, M_p, marker='o')
plt.xlabel('periodic of exoplanet  [year] log')
plt.ylabel('planetary mass [Mj] log')
plt.title(detector+' 5 mHz T = 4 year Mb = 1 Msun')
plt.axvline(x=0, linestyle=":", color="m")
plt.show(block=True)
print(P, M_p)

# # 准备数据
# dataR_x = pd.DataFrame(R_x)  # 关键1，将ndarray格式转换为DataFrame
# dataM_p_y = pd.DataFrame(M_p_y)
#
# writer = pd.ExcelWriter('taiji.xlsx')  # 关键2，创建名称为hhh的excel表格
# dataR_x.to_excel(writer, 'page_1', float_format='%.5f')  # 关键3，float_format 控制精度，将data_df写到hhh表格的第一页中。
# 若多个文件，可以在page_2中写入
# dataM_p_y.to_excel(writer, 'page_2', float_format='%.5f')
# writer.save()
