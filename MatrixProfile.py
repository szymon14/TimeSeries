import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from MatrixProfileAlgorithm import matrix_profile
from Utils import z_score

def analiza_szeregu_mp(m, axs):
    mp_lista = matrix_profile(m, df_clean, 'Wypadki_Drogowe', 'Wypadki_Drogowe')
    matrix_prof_dist = np.array(mp_lista)

    srednia_odleglosc_mp = matrix_prof_dist.mean()

    print("Średnia odległość między najbliższymi motywami (MP):", srednia_odleglosc_mp)

    szereg_scaled = z_score(szereg)
    najwiekszy_idx = np.argmax(szereg_scaled)

    axs[0].plot(szereg_scaled, label="Wypadki drogowe (Z-score)", color='royalblue', alpha=0.7)
    axs[0].legend(loc='upper right')
    axs[0].grid(True, linestyle='--', alpha=0.5)

    axs[1].plot(matrix_prof_dist, color='crimson', linewidth=2, label='Matrix Profile Distance')
    axs[1].axhline(srednia_odleglosc_mp, color='black', linestyle=':', alpha=0.8,
                   label=f'Średnia MP: {srednia_odleglosc_mp:.2f}')

    # Znajdujemy i zaznaczamy największą anomalię (najwyższy szczyt na profilu)
    najwyższy_szczyt_idx = np.argmax(matrix_prof_dist)
    axs[1].plot(najwyższy_szczyt_idx, matrix_prof_dist[najwyższy_szczyt_idx], 'ro', markersize=10,
                label='Największa anomalia (Discord)')

    axs[1].set_title(f'Cross-Matrix Profile (Długość wzorca m = {m} dni)', fontsize=14, fontweight='bold')
    axs[1].set_ylabel('Odległość do najbliższego motywu', fontsize=12)
    axs[1].set_xlabel('Indeks dnia w szeregu czasowym', fontsize=12)
    axs[1].legend(loc='upper right')
    axs[1].grid(True, linestyle='--', alpha=0.5)
    return najwiekszy_idx

df =  pd.read_csv('policja_info.csv')
df['Data'] = pd.to_datetime(df['Data'])
df = df.sort_values('Data')

df_clean = df[['Wypadki_Drogowe','Data']].dropna()
szereg = df_clean['Wypadki_Drogowe'].astype(float).values

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(16, 22), constrained_layout=True)
idx7 = analiza_szeregu_mp(7, axes[0:2])
idx30 = analiza_szeregu_mp(30, axes[2:4])
print("Anomalia m=7:\n", df_clean.iloc[idx7])
print("Anomalia m=7:\n", df_clean.iloc[idx7]) # na z-score mamy odległość blisko 6 czyli aż 6 odchyleń standardowych od średniej na matrix profile mamy odległość 4 czyli odległość od najbliższego elementu
plt.show()
######### górki dla 30 dni są dłuższe bo dłużej pamięta anomalie