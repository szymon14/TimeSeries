import math
import numpy as np

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

def z_score(series): # Amplitude scaling
    return (series - np.mean(series)) / np.std(series)


def evaluate_series(title, s1, s2):
    dist = calculate_distance(s1, s2)
    rmse = calculate_rmse(s1, s2)

    print(f"\n********* {title} ************")
    print(f"Odległość Euklidesowa: {dist:.4f}")
    print(f"RMSE:                  {rmse:.4f}")
    return dist, rmse