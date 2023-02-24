#!/usr/bin/env python
# coding: utf-8

# ### UCI 전복
# 
# https://archive.ics.uci.edu/ml/datasets/abalone
# 
# data url: https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data
# 
# 1. [함수] requests 패키지를 이용해 데이터 가져와서 ndarray로 변환
# 
# 2. [함수] 성별이 'M'인 데이터를 필터, Length 와 Diameter 간 상관도를 반환
# 
# 3. __name__ 값이 __main__이면, 1,2 함수를 실행, 2번함수의 반환값을 프린트
# 
# bonus : sqlite3 데이터베이스 생성.
# 
# 참고 : https://docs.python.org/3/library/sqlite3.html
# 
# 테이블 DDL
# 
#     create table abalone(
#         length number,
#         diameter number);

# In[1]:


import requests
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def fetch_uci_data(s):
    d = requests.get(url)
    t = []
    for line in d.text.split('\n'):
        if len(line) != 0:
            t.append(line.split(','))
        
    #len_check = set()
    #for e in t:
    #    if(len(e) == 1):
    #        print(e)
    #        len_check.add(len(e))
    #print(len_check)
    
    return np.array(t)

def get_corr(d):
    male_filter = d[:, 0] == 'M'
    male_sampels = d[male_filter]
    male_length = male_sampels[:, 1].astype(np.float64)
    male_diameter = male_sampels[:, 2].astype(np.float64)
    return np.corrcoef(male_length, male_diameter)


# In[2]:


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data'

if __name__ == '__main__':
    np_data = fetch_uci_data(url)
    corr_value = get_corr(np_data)
    print(corr_value[0, 1])


