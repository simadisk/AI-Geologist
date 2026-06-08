# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 13:46:13 2026

@author: User
"""

import torch
import torch.nn as nn
import numpy as np
import io
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response

# =====================================================================
# 1. ΤΟ ΑΡΧΙΤΕΚΤΟΝΙΚΟ ΣΧΕΔΙΟ ΤΟΥ ΜΟΝΤΕΛΟΥ (Ακριβώς όπως στο Colab)
# =====================================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): 
        return self.conv(x)

class AttentionBlock3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, kernel_size=1), nn.BatchNorm3d(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, kernel_size=1), nn.BatchNorm3d(F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, kernel_size=1), nn.BatchNorm3d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        return x * self.psi(self.relu(self.W_g(g) + self.W_x(x)))

class AttentionUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=6):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 16); self.pool1 = nn.MaxPool3d(2)
        self.down2 = DoubleConv(16, 32); self.pool2 = nn.MaxPool3d(2)
        self.down3 = DoubleConv(32, 64); self.pool3 = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv(64, 128)

        self.upconv3 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.att3 = AttentionBlock3D(64, 64, 32); self.up3 = DoubleConv(128, 64)

        self.upconv2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.att2 = AttentionBlock3D(32, 32, 16); self.up2 = DoubleConv(64, 32)

        self.upconv1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.att1 = AttentionBlock3D(16, 16, 8); self.up1 = DoubleConv(32, 16)

        self.outc = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool1(x1))
        x3 = self.down3(self.pool2(x2))
        x4 = self.bottleneck(self.pool3(x3))

        g3 = self.upconv3(x4); x = self.up3(torch.cat([g3, self.att3(g3, x3)], dim=1))
        g2 = self.upconv2(x);  x = self.up2(torch.cat([g2, self.att2(g2, x2)], dim=1))
        g1 = self.upconv1(x);  x = self.up1(torch.cat([g1, self.att1(g1, x1)], dim=1))
        
        return self.outc(x)

# =====================================================================
# 2. FASTAPI SERVER & ΦΟΡΤΩΣΗ ΤΟΥ ΕΓΚΕΦΑΛΟΥ
# =====================================================================
app = FastAPI(title="Seismic AI API")

print("⏳ Φόρτωση του εγκεφάλου AI στη μνήμη. Παρακαλώ περιμένετε...")

# Βρίσκουμε αν υπάρχει κάρτα γραφικών ή αν θα τρέξει στον επεξεργαστή (CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Φτιάχνουμε το άδειο "αυτοκίνητο"
model = AttentionUNet3D(in_channels=1, out_channels=6).to(device)

# Φορτώνουμε τον "οδηγό" (.pth αρχείο) μέσα στο αυτοκίνητο
model.load_state_dict(torch.load("model/SOTA_BEST_seismic_model.pth", map_location=device))
model.eval() # Κλειδώνουμε το μοντέλο ώστε να μην εκπαιδεύεται πια

print(f"✅ Το μοντέλο φορτώθηκε με επιτυχία στο {device}!")



# =====================================================================
# 3. ΤΟ ΚΑΝΑΛΙ ΕΠΙΚΟΙΝΩΝΙΑΣ (ENDPOINT)
# =====================================================================
@app.get("/")
def home():
    return {"status": "Online", "message": "Ο AI Γεωλόγος είναι στη θέση του και περιμένει δεδομένα!"}


# =====================================================================
# 4. Η ΠΥΛΗ ΠΡΟΒΛΕΨΗΣ (INFERENCE ENDPOINT)
# =====================================================================
@app.post("/predict")
async def predict_seismic(file: UploadFile = File(...)):
    print("📥 Λήψη νέου σεισμικού τμήματος...")
    
    # 1. Διαβάζουμε το αρχείο που μας έστειλε ο χρήστης
    contents = await file.read()
    patch = np.load(io.BytesIO(contents))
    
    # 2. Το προετοιμάζουμε για το PyTorch
    patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. Το περνάμε από το μοντέλο (Inference)
    with torch.no_grad():
        output = model(patch_tensor)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        
    print("✅ Η πρόβλεψη ολοκληρώθηκε! Αποστολή αποτελέσματος...")
    
    # 4. Πακετάρουμε το αποτέλεσμα και το στέλνουμε πίσω
    out_io = io.BytesIO()
    np.save(out_io, pred)
    out_io.seek(0)
    
    return Response(content=out_io.read(), media_type="application/octet-stream")









