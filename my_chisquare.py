import math
import numpy as np

# 自己实现chisquare,参数为两个列表obs,exp，返回为包含卡方值和p值的列表
#eg:obs=[8,7,7] ,exp=[8,8,8]
#return [0.25, 0.8824969025845955]
def my_chisquare(obs, exp):
    # 将列表转化为numpy.ndarray类型
    obs = np.atleast_1d(np.asanyarray(obs))
    exp = np.atleast_1d(np.asanyarray(exp))
    if obs.size != exp.size:
        print('The size of the obs and the exp  array is not equal')
        exit()

    # 得到ndarray类型，得到各项的理论与观察的相对偏离距离，相加即为卡方值
    terms = (obs - exp) ** 2 / exp
    # 求得卡方值,numpy.float64类型
    stat = terms.sum(axis=0)
    # 计算obs,exp的维度
    num_obs = terms.size
    # 调用自己写的求p值的函数，得到p值
    p = chisqr2pValue(num_obs - 1, stat)
    chisquare_list = []
    chisquare_list.append(stat)
    chisquare_list.append(p)
    return chisquare_list

# 用斯特林公式来近似求伽马函数（卡方检验）
def getApproxGamma(n):
    RECIP_E = 0.36787944117144232159552377016147
    TWOPI = 6.283185307179586476925286766559
    d = 1.0 / (10.0 * n)
    d = 1.0 / ((12 * n) - d)
    d = (d + n) * RECIP_E
    d = math.pow(d, n)
    d = d * math.sqrt(TWOPI / n)
    return d


# 不完全伽马函数中需要调用的函数（卡方检验）
def KM(s, z):
    _sum = 1.0
    log_nom = math.log(1.0)
    log_denom = math.log(1.0)
    log_s = math.log(s)
    log_z = math.log(z)
    for i in range(1000):
        log_nom += log_z
        s = s + 1
        log_s = math.log(s)
        log_denom += log_s
        log_sum = log_nom - log_denom
        log_sum = float(log_sum)
        _sum += math.exp(log_sum)

    return _sum


# 不完全伽马函数，采用计算其log值（卡方检验）
def log_igf(s, z):
    if z < 0.0:
        return 0.0
    sc = float((math.log(z) * s) - z - math.log(s))
    k = float(KM(s, z))
    return math.log(k) + sc

#卡方检验求p值
# dof是自由度，chi_squared为卡方值，该函数实现知道自由度和卡方值求p值
# 核心是用不完全伽马函数除以伽马函数，两者都采用近似函数求解
# 参见https://blog.csdn.net/idatamining/article/details/8565042
def chisqr2pValue(dof, chi_squared):
    dof = int(dof)
    chi_squared = float(chi_squared)
    if dof < 1 or chi_squared < 0:
        return 0.0
    k = float(dof) * 0.5
    v = chi_squared * 0.5
    # 自由度为2时
    if dof == 2:
        return math.exp(-1.0 * v)
    # 不完全伽马函数，采用计算其log值
    incompleteGamma = log_igf(k, v)
    # 如果过小或者无穷
    if math.exp(incompleteGamma) <= 1e-8 or math.exp(incompleteGamma) == float('inf'):
        return 1e-14

    # 完全伽马函数，用斯特林公式近似
    gamma = float(math.log(getApproxGamma(k)))
    incompleteGamma = incompleteGamma - gamma
    if math.exp(incompleteGamma) > 1:
        return 1e-14
    pvalue = float(1.0 - math.exp(incompleteGamma))
    return pvalue


obs1 = [8, 7, 7]
exp1 = [8, 8,0]
lst = my_chisquare(obs1, exp1)
print(lst)