import pandas as pd
from scipy import interpolate
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt

raw_acc = pd.read_csv('data/E4AccData.csv', usecols = ['Time', 'Acc_X', 'Acc_Y', 'Acc_Z'])
raw_gsr = pd.read_csv('data/E4GsrData.csv', usecols = ['Time', 'GSR'])
raw_tmp = pd.read_csv('data/E4TmpData.csv', usecols = ['Time', 'Tmp'])
raw_sk = pd.read_csv('data/SkeletonData.csv', skiprows=[0], header=None)
raw_lv = pd.read_csv('data/LabelingData.csv', usecols = ['Time', 'Labeling'])

def preprocess():
    """
    1. 각 데이터를 0.03125초 간격으로 재구성 (Skeleton data는 부정확한 값 제거 후 재구성)
    2. 각 데이터 정규분포로 변환
    3. 데이터프레임 통합 후 csv 파일로 저장
    """
    sk_time = raw_sk.iloc[:, 0]

    starttime = max(raw_acc['Time'][0], raw_gsr['Time'][0], raw_tmp['Time'][0], raw_lb['Time'][0])
    endtime = min(raw_acc['Time'][len(raw_acc) - 1], raw_gsr['Time'][len(raw_gsr) - 1], raw_tmp['Time'][len(raw_tmp) - 1], raw_lb['Time'][len(raw_tmp) - 1])

    start = sk_time > starttime
    end = sk_time < endtime

    time = sk_time.loc[start & end]

    # ACC data
    acc_time = raw_acc['Time']
    raw_acc_x = raw_acc['Acc_X']
    raw_acc_y = raw_acc['Acc_Y']
    raw_acc_z = raw_acc['Acc_Z']

    acc_xf = interpolate.interp1d(acc_time, raw_acc_x)
    acc_yf = interpolate.interp1d(acc_time, raw_acc_y)
    acc_zf = interpolate.interp1d(acc_time, raw_acc_z)

    acc_x = acc_xf(time)
    acc_y = acc_yf(time)
    acc_z = acc_zf(time)

    acc_x = ss.zscore(acc_x)
    acc_y = ss.zscore(acc_y)
    acc_z = ss.zscore(acc_z)

    final_acc = pd.DataFrame({'Time': time, 'Acc_X': acc_x, 'Acc_Y': acc_y, 'Acc_Z': acc_z})


    # GSR data
    gsr_time = raw_gsr['Time']
    raw_gsr2 = raw_gsr['GSR']
    gsr_f = interpolate.interp1d(gsr_time, raw_gsr2)
    gsr = gsr_f(time)
    gsr = ss.zscore(gsr)
    final_gsr = pd.DataFrame({'Time': time, 'GSR': gsr})


    # Tmp data
    tmp_time = raw_tmp['Time']
    raw_tmp2 = raw_tmp['Tmp']
    tmp_f = interpolate.interp1d(tmp_time, raw_tmp2)
    tmp = tmp_f(time)
    tmp = ss.zscore(tmp)
    final_tmp = pd.DataFrame({'Time': time, 'Tmp': tmp})


    # Skeleton data
    """
    - Confidence 0.5 이하 데이터 난수 표시
    """
    final_sk = pd.DataFrame({'Time': time})

    for i in range(0, 19):
        df = pd.DataFrame()
        df['X_{}'.format(i + 1)] = raw_sk.iloc[:, 1 + 4 * i]
        df['Y_{}'.format(i + 1)] = raw_sk.iloc[:, 2 + 4 * i]
        df['Z_{}'.format(i + 1)] = raw_sk.iloc[:, 3 + 4 * i]
        df['Confidence'] = raw_sk.iloc[:, 4 + 4 * i]

        under50 = df['Confidence'] < 0.5
        df.loc[under50] = np.nan

        sk_x = ss.zscore(df['X_{}'.format(i+1)], nan_policy='omit')
        sk_y = ss.zscore(df['Y_{}'.format(i+1)], nan_policy='omit')
        sk_z = ss.zscore(df['Z_{}'.format(i+1)], nan_policy='omit')

        final_sk['X_{}'.format(i+1)] = sk_x
        final_sk['Y_{}'.format(i+1)] = sk_y
        final_sk['Z_{}'.format(i+1)] = sk_z


    # Labeling data
    lev_time = raw_lv['Time']
    raw_lv2 = raw_lv['Labeling']
    lv_f = interpolate.interp1d(lev_time, raw_lv2)
    lv = lv_f(time)
    lv = ss.zscore(lv)
    final_lv = pd.DataFrame({'Time': time, 'Engagement': lv})

    # Data 모두 하나의 csv 파일로 통합
    merge1 = pd.merge(final_acc, final_gsr, on='Time')
    merge2 = pd.merge(final_tmp, final_sk, on='Time')
    merge3 = pd.merge(merge2, final_lv, on='Time')
    merge = pd.merge(merge1, merge3, on='Time')

    merge.to_csv('data/ProcessedData.csv', index=False)

def pre_plus_success():
    raw_suc = pd.read_csv('data/{success}.csv', usecols=['Time', 'Labeling'])
    other = pd.read_csv('data/ProcessedData.csv')

    t_time = other.iloc[:, 0]
    starttime = raw_suc['Time'][0]
    endtime = raw_suc['Time'][len(raw_acc) - 1]

    start = t_time > starttime
    end = t_time < endtime
    time = t_time.loc[start & end]

    suc_time = raw_suc['Time']
    raw_suc2 = raw_suc['Labeling']
    suc_f = interpolate.interp1d(suc_time, raw_suc2)
    suc = suc_f(time)
    suc = ss.zscore(suc)
    final_suc = pd.DataFrame({'Time': time, 'Success': suc})

    merge = pd.merge(final_suc, other, on='Time')
    merge.to_csv('data/ProcessedData2.csv', index=False)

def sensor2img(interval, stride):

    sensors = pd.read_csv('data/ProcessedData.csv')
    starttime = sensors['Time'][0]
    endtime = sensors['Time'][len(raw_acc) - 1]

    time = sensors['Time']
    cnt = starttime

    while (cnt + interval) < endtime:
        ub = time > cnt
        lb = time < (cnt + interval)
        time = time.loc[ub & lb]
        """
        plt.plot(time, y-axis value, color)
        plt.savefig('location/name.png')
        """
        cnt += stride