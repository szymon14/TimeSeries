import numpy as np
import pandas as pd
import math
from scipy.signal import medfilt
from fastdtw import fastdtw


def calculate_distance(szereg1, szereg2):
    sum = 0
    n = min(len(szereg1),len(szereg2))
    for i in range(n):
        sum += (szereg1[i] - szereg2[i])**2
    return math.sqrt(sum)

def calculate_rmse(szereg1, szereg2): # Pierwiastek błędu sredniokwadratowego
    sum = 0
    n = min(len(szereg1),len(szereg2))
    for i in range(n):
        sum += (szereg1[i] - szereg2[i])**2
    return math.sqrt(sum / n)

data = pd.read_csv('policja_info.csv', delimiter=',')
data['Data'] = pd.to_datetime(data['Data'])
data = data.sort_values('Data')

df_clean = data[['Wypadki_Drogowe','Ranni']].dropna()

szereg_2 = df_clean['Wypadki_Drogowe'].values
szereg_3 = df_clean['Ranni'].values

odleglosc = calculate_distance(szereg_2, szereg_3)
srednia_odleglosc = calculate_rmse(szereg_2, szereg_3)
print(szereg_2)
print(szereg_3)

print('********* Euclidian Distance ************')
print(odleglosc) # Nic mądrego to nie mówi
print(srednia_odleglosc)

print('********* Amplitude Scaling ************')
######### Teraz robię Amplitude Scaling Z-SCORE
szereg_2_ascal = ((df_clean['Wypadki_Drogowe'] - df_clean['Wypadki_Drogowe'].mean())/df_clean['Wypadki_Drogowe'].std()).values
szereg_3_ascal = ((df_clean['Ranni'] - df_clean['Ranni'].mean())/df_clean['Ranni'].std()).values

odleglosc_ascal = calculate_distance(szereg_2_ascal, szereg_3_ascal)
srednia_odleglosc_ascal = calculate_rmse(szereg_2_ascal, szereg_3_ascal)

print(odleglosc_ascal) # Wyniki są naprawdę dobre
print(srednia_odleglosc_ascal)

##################################Trend średnią kroczącą
print('********* Linear Trend ************')
szereg_2_trend = df_clean['Wypadki_Drogowe'].rolling(window=30, min_periods=1).mean()
szereg_3_trend = df_clean['Ranni'].rolling(window=30, min_periods=1).mean()

odleglosc_trend = calculate_distance(szereg_2_trend, szereg_3_trend)
srednia_odleglosc_trend = calculate_rmse(szereg_2_trend, szereg_3_trend)

print(odleglosc_trend) # Te tylko trochę lepsze od zwykłej odległosći Euklidesowej
print(srednia_odleglosc_trend)

##################################Trend + ASCAL
print('********* Linear Trend + ASCAL  ************')
szereg_2_trend_ascal = ((szereg_2_trend - szereg_2_trend.mean())/szereg_2_trend.std()).values
szereg_3_trend_ascal = ((szereg_3_trend - szereg_3_trend.mean())/szereg_3_trend.std()).values

odleglosc_trend_ascal = calculate_distance(szereg_2_trend_ascal, szereg_3_trend_ascal)
srednia_odleglosc_trend_ascal = calculate_rmse(szereg_2_trend_ascal, szereg_3_trend_ascal)

print(odleglosc_trend_ascal) # Jeszcze bardziej poprawiono wynik
print(srednia_odleglosc_trend_ascal)

print('********* Usuwanie szumu (Median Filter / R smooth) + ASCAL ************') # ten co w r odpowiada smooth()
window_size = 31 # nie moze byc za duże sprawdzilem 365 dni i był dobry wynik ale czy to ma sens
szereg_2_smooth = medfilt(df_clean['Wypadki_Drogowe'], kernel_size=window_size)
szereg_3_smooth = medfilt(df_clean['Ranni'], kernel_size=window_size)

szereg_2_smooth_ascal = (szereg_2_smooth - np.mean(szereg_2_smooth)) / np.std(szereg_2_smooth)
szereg_3_smooth_ascal = (szereg_3_smooth - np.mean(szereg_3_smooth)) / np.std(szereg_3_smooth)

odleglosc_smooth_ascal = calculate_distance(szereg_2_smooth_ascal, szereg_3_smooth_ascal)
srednia_odleglosc_smooth_ascal = calculate_rmse(szereg_2_smooth_ascal, szereg_3_smooth_ascal)

print(odleglosc_smooth_ascal)## jest gorzej
print(srednia_odleglosc_smooth_ascal)

print('************DTW*********')
n = min(len(szereg_2_ascal),len(szereg_3_ascal))
dist_func = lambda x, y: abs(x - y) # z racji jednego wymiaru to jest to samo co odległość euklidesowa
dtw_raw_distance, _ = fastdtw(szereg_2_ascal, szereg_3_ascal, dist=dist_func)
print(f"DTW dla Z-Score: {dtw_raw_distance:.4f}")
print(f"Dzielimy przez długość szeregu: {dtw_raw_distance/n:.4f}")

print('************ Pierwsza pochodna + ASCAL *********')

szereg_2_diff_raw = df_clean['Wypadki_Drogowe'].diff()
szereg_3_diff_raw = df_clean['Ranni'].diff()

szereg_2_diff = ((szereg_2_diff_raw - szereg_2_diff_raw.mean()) / szereg_2_diff_raw.std()).dropna().values
szereg_3_diff = ((szereg_3_diff_raw - szereg_3_diff_raw.mean()) / szereg_3_diff_raw.std()).dropna().values

odleglosc_diff = calculate_distance(szereg_2_diff, szereg_3_diff)
srednia_odleglosc_diff = calculate_rmse(szereg_2_diff, szereg_3_diff)

print(odleglosc_diff)## Kiepsko
print(srednia_odleglosc_diff)