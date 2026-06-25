import numpy as np

def matrix_profile(m,df_clean,str1,str2):
    s1 = df_clean[str1].astype(float).values
    s2 = df_clean[str2].astype(float).values
    s1_scaled = (s1 - s1.mean()) / s1.std()
    s2_scaled = (s2 - s2.mean()) / s2.std()
    n = min(len(s2_scaled),len(s1_scaled))

    matrix_profile_distances = []
    for i in range(n - m+1):
        subseq = s1_scaled[i:i+m] # kawałek od wielkosi m zaczynajacy sie od indeksu i
        min_dist = float('inf')
        for j in range(n - m+1):
            target = s2_scaled[j:j+m]
            distance = np.sqrt(np.sum((subseq - target)**2))
            if distance < min_dist:
                min_dist = distance
        matrix_profile_distances.append(min_dist)
    return matrix_profile_distances
