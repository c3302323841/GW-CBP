import constant as con

f = 0.01 / con.T_dimension  # 引力波频率（10 mHz）s-1
T_obs = 4 * 31536000 * con.T_dimension  # 观测周期
M_b = 1 * con.M_sun

na = 14  # x轴的P的数量
nb = 1000  # 寻找最小Mp的上限
nc = 1
# nc: tianqin is 1, taiji ji 2, lisa is 3, tianqin + taiji is 4 tianqin + lisa is 5,
# lisa +taiji is 6, three is 7
