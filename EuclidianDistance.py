import pandas as pd
import numpy as np
from scipy.signal import medfilt
from fastdtw import fastdtw
from Utils import evaluate_series, z_score, calculate_distance
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('policja_info.csv', delimiter=',')
data['Data'] = pd.to_datetime(data['Data'])
data = data.sort_values('Data')

df_clean = data[['Wypadki_Drogowe','Ranni']].dropna()

szereg_2 = df_clean['Wypadki_Drogowe'].values
szereg_3 = df_clean['Ranni'].values

# Nic mądrego to nie mówi
evaluate_series("Euclidian Distance", szereg_2, szereg_3)

######### Teraz robię Amplitude Scaling Z-SCORE
szereg_2_ascal = z_score(df_clean["Wypadki_Drogowe"]).values
szereg_3_ascal = z_score(df_clean["Ranni"]).values

# Wyniki są naprawdę dobre
evaluate_series("Amplitude Scaling", szereg_2_ascal, szereg_3_ascal)

##################################Trend średnią kroczącą
szereg_2_trend = (
    df_clean["Wypadki_Drogowe"].rolling(window=30, min_periods=1).mean()
)
szereg_3_trend = df_clean["Ranni"].rolling(window=30, min_periods=1).mean()

# Te tylko trochę lepsze od zwykłej odległosći Euklidesowej
evaluate_series("Linear Trend", szereg_2_trend.values, szereg_3_trend.values)

##################################Trend + ASCAL
szereg_2_trend_ascal = z_score(szereg_2_trend).values
szereg_3_trend_ascal = z_score(szereg_3_trend).values

# Jeszcze bardziej poprawiono wynik
evaluate_series(
    "Linear Trend + ASCAL", szereg_2_trend_ascal, szereg_3_trend_ascal
)


# ten co w r odpowiada smooth() czyli usuwanie szumu
# nie moze byc za duże sprawdzilem 365 dni i był dobry wynik ale czy to ma sens
window_size = 31
szereg_2_smooth = medfilt(df_clean["Wypadki_Drogowe"], kernel_size=window_size)
szereg_3_smooth = medfilt(df_clean["Ranni"], kernel_size=window_size)

szereg_2_smooth_ascal = z_score(szereg_2_smooth)
szereg_3_smooth_ascal = z_score(szereg_3_smooth)

# jest gorzej
evaluate_series(
    "Usuwanie szumu (Median Filter / R smooth) + ASCAL",
    szereg_2_smooth_ascal,
    szereg_3_smooth_ascal,
)

# DTW
print("\n************DTW*********")
n = min(len(szereg_2_ascal), len(szereg_3_ascal))
dist_func = (
    lambda x, y: abs(x - y)
)  # z racji jednego wymiaru to jest to samo co odległość euklidesowa
dtw_raw_distance, _ = fastdtw(szereg_2_ascal, szereg_3_ascal, dist=dist_func)
print(f"DTW dla Z-Score: {dtw_raw_distance:.4f}")
print(f"Dzielimy przez długość szeregu: {dtw_raw_distance/n:.4f}")


szereg_2_diff_raw = df_clean["Wypadki_Drogowe"].diff()
szereg_3_diff_raw = df_clean["Ranni"].diff()

szereg_2_diff = z_score(szereg_2_diff_raw.dropna()).values
szereg_3_diff = z_score(szereg_3_diff_raw.dropna()).values

# Kiepsko
evaluate_series("Pierwsza pochodna + ASCAL", szereg_2_diff, szereg_3_diff)


szereg_glowny = df_clean["Wypadki_Drogowe"].values

# Tniemy na roczne
dlugosc_okna = 365
liczba_szeregow = 11
szeregi = []

for i in range(liczba_szeregow):
    start = i * dlugosc_okna
    end = start + dlugosc_okna
    szeregi.append(szereg_glowny[start:end])

szeregi_clean = [z_score(s) for s in szeregi]

dist_raw = np.zeros((liczba_szeregow, liczba_szeregow))
dist_clean = np.zeros((liczba_szeregow, liczba_szeregow))
dist_dtw = np.zeros((liczba_szeregow, liczba_szeregow))

for i in range(liczba_szeregow):
    for j in range(liczba_szeregow):
        # surowa euklidesowa
        dist_raw[i, j] = calculate_distance(szeregi[i], szeregi[j])

        # po usunięciu zniekształceń (Z-score)
        dist_clean[i, j] = calculate_distance(szeregi_clean[i], szeregi_clean[j])

        # DTW
        dist_dtw[i, j], _ = fastdtw(szeregi_clean[i], szeregi_clean[j], dist=lambda x, y: abs(x - y))

linkage_raw = linkage(squareform(dist_raw), method="ward")
linkage_clean = linkage(squareform(dist_clean), method="ward")
linkage_dtw = linkage(squareform(dist_dtw), method="ward")

# Rysowanie 3 dendrogramów obok siebie
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

dendrogram(linkage_raw, ax=axes[0])
axes[0].set_title("1) Surowe dane (Euklides)")

dendrogram(linkage_clean, ax=axes[1])
axes[1].set_title("2) Po Z-Score (Euklides)")

dendrogram(linkage_dtw, ax=axes[2])
axes[2].set_title("3) DTW")

plt.show()

### Ekstrakcja cech
features_list = []

# Pętla po oryginalnych szeregach
for s in szeregi:
    features = [
        np.mean(s),  # 1. Średnia
        np.std(s),  # 2. Odchylenie standardowe
        np.var(s),  # 3. Wariancja
        np.min(s),  # 4. Minimum
        np.max(s),  # 5. Maximum
        np.ptp(s),  # 6. Zakres (max - min)
        np.median(s),  # 7. Mediana
        np.quantile(s, 0.25),  # 8. Kwartyl 1
        np.quantile(s, 0.75),  # 9. Kwartyl 3
        np.sum(s),  # 10. Suma wartości
        np.mean(np.diff(s)),  # 11. Średnia z 1. pochodnej
        np.std(np.diff(s)),  # 12. Odchylenie z 1. pochodnej
        np.max(np.diff(s)),  # 13. Maksimum z 1. pochodnej
        np.min(np.diff(s)),  # 14. Minimum z 1. pochodnej
    ]
    features_list.append(features)

#macierz X_features ma wymiary 11 x 14
X_features = np.array(features_list)

# standaryzacja cech potrzebna bo mają różne skale i jednostki
X_features_scaled = StandardScaler().fit_transform(X_features)

# generowanie połączeń dla macierzy cech
linkage_features = linkage(X_features_scaled, method="single")

# Wizualizacja macierzy odległości dla cech
plt.figure(figsize=(8, 6))
plt.imshow(squareform(pdist(X_features_scaled)), cmap='viridis')
plt.colorbar(label='Odległość Euklidesowa')
plt.title("Macierz odległości między szeregami (Features)")
plt.show()


# rysowanie dendrogramu dla wybranych cech
plt.figure(figsize=(8, 6))
dendrogram(linkage_features)
plt.title("Klasteryzacja na podstawie wyekstrahowanych cech")
plt.xlabel("Indeks szeregu")
plt.ylabel("Odległość")
plt.tight_layout()
plt.show()