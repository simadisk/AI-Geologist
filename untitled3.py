# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:51:17 2026

@author: User
"""

import numpy as np

# 1. Βάλε το σωστό μονοπάτι για το μεγάλο αρχείο
big_cube_path = r"C:\Users\User\OneDrive\Desktop\Project\data\f3_seismic.npy"

print("Φόρτωση μεγάλου κύβου...")
big_cube = np.load(big_cube_path)

# 2. Κόβουμε ένα μικρό κομμάτι 64x64x64 από τη μέση περίπου του χάρτη
test_patch = big_cube[100:164, 100:164, 100:164]

# 3. Το αποθηκεύουμε στον φάκελο του Project
np.save(r"C:\Users\User\OneDrive\Desktop\Project\test_patch.npy", test_patch)
print("✅ Το αρχείο test_patch.npy δημιουργήθηκε με επιτυχία!")