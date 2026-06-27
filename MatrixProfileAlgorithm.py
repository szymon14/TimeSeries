import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def matrix_profile(m, df_clean, str1, str2):
    s1 = df_clean[str1].astype(float).values
    s2 = df_clean[str2].astype(float).values
    s1_scaled = (s1 - s1.mean()) / s1.std()
    s2_scaled = (s2 - s2.mean()) / s2.std()
    
    n = min(len(s2_scaled), len(s1_scaled))
    s1_scaled, s2_scaled = s1_scaled[:n], s2_scaled[:n]

    # tworzy widok wszystkih możliwych okien z szeregu s2 o długości m
    windows2 = sliding_window_view(s2_scaled, m)

    matrix_profile_distances = []
    for i in range(n - m + 1):
        subseq = s1_scaled[i:i+m]
        # odległość euklidesowa
        distances = np.linalg.norm(windows2 - subseq, axis=1)
        # zbyt małe odległości zmieniamy na inf
        distances[max(0, i - m + 1) : i + m] = np.inf
        matrix_profile_distances.append(np.min(distances))

    return matrix_profile_distances
