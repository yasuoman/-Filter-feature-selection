import pandas as pd
import numpy as np
#标签只能为0和1,样本空间任意，
#samle:n*m 的列表
#label:m*1 的列表
#返回每个特征的fisher score值的一个列表
#eg:  sample = [[1,2,3],[1,0,1],[1,5,6]]
#     label = [1, 0, 1]
#return  lst=[nan, 1.8148148148148149, 1.8148148148148149]
def binary_fisher_score(sample,label):

    if len(sample) != len(label):
        print('Sample does not match label')
        exit()
    df1 = pd.DataFrame(sample)
    df2 = pd.DataFrame(label, columns=['label'])
    data = pd.concat([df1, df2], axis=1)  # 合并成为一个dataframe

    data0 = data[data.label == 0]#对标签分类，分成包含0和1的两个dataframe
    data1 = data[data.label == 1]
    n = len(label)#标签长度
    n1 = sum(label)#1类标签的个数
    n0 = n - n1#0类标签的个数
    lst = []#用于返回的列表
    features_list = list(data.columns)[:-1]
    for feature in features_list:

        # 算关于data0
        m0_feature_mean = data0[feature].mean()  # 0类标签在第m维上的均值
        # 0类在第m维上的sw
        m0_SW=sum((data0[feature] -m0_feature_mean )**2)
        # 算关于data1
        m1_feature_mean = data1[feature].mean()  # 1类标签在第m维上的均值
        # 1类在第m维上的sw
        m1_SW=sum((data1[feature] -m1_feature_mean )**2)
        # 算关于data
        m_all_feature_mean = data[feature].mean()  # 所有类标签在第m维上的均值

        m0_SB = n0 / n * (m0_feature_mean - m_all_feature_mean) ** 2
        m1_SB = n1 / n * (m1_feature_mean - m_all_feature_mean) ** 2
        #计算SB
        m_SB = m1_SB + m0_SB
        #计算SW
        m_SW = (m0_SW + m1_SW) / n
        if m_SW == 0:
            # 0/0类型也是返回nan
            m_fisher_score = np.nan
        else:
            # 计算Fisher score
            m_fisher_score = m_SB / m_SW
        #Fisher score值添加进列表
        lst.append(m_fisher_score)

    return lst

sample = [[1,2,3],[1,0,1],[1,5,6]]
label = [1, 0, 1]
print(binary_fisher_score(sample,label))
