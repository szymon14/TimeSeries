import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from MatrixProfileAlgorithm import matrix_profile

fig, axs = plt.subplots(2,1,sharex=True, figsize = (16,10))

df =  pd.read_csv('policja_info.csv')
df['Data'] = pd.to_datetime(df['Data'])
df = df.sort_values('Data')

df_clean = df[['Wypadki_Drogowe','Ranni']].dropna()

szereg_2 = df_clean['Wypadki_Drogowe'].astype(float).values
szereg_3 = df_clean['Ranni'].astype(float).values

m=30
mp_lista = matrix_profile(m,df_clean,'Wypadki_Drogowe','Ranni')
matrix_prof_dist = np.array(mp_lista)

srednia_odleglosc_mp = matrix_prof_dist.mean()

print("Średnia odległość między najbliższymi motywami (MP):", srednia_odleglosc_mp)

szereg_2_scaled = (szereg_2 - szereg_2.mean())/szereg_2.std()
szereg_3_scaled = (szereg_3 - szereg_3.mean())/szereg_3.std()

axs[0].plot(szereg_2_scaled,label = "Wypadki drogowe (Z-score)", color = 'royalblue', alpha=0.7)
axs[0].plot(szereg_3_scaled, label='Ranni (Z-score)', color='orange', alpha=0.7)
axs[0].set_title('Porównanie Szeregów Czasowych', fontsize=14, fontweight='bold')
axs[0].set_ylabel('Skalowana wartość', fontsize=12)
axs[0].legend(loc='upper left')
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
axs[1].legend(loc='upper left')
axs[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()