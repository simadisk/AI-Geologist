# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 14:08:28 2026

@author: User
"""


import plotly.graph_objects as go
import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import io

# 1. Ορισμός των Γεωλογικών Κλάσεων (F3 Dataset)
CLASS_NAMES = {
    0: "Clay / Others",
    1: "Carbonates",
    2: "Salt",
    3: "Sandstone",
    4: "Shale",
    5: "Tuff / Basement"
}

# Ρυθμίσεις Σελίδας
st.set_page_config(page_title="AI Γεωλόγος Pro", page_icon="🌍", layout="wide")

st.title("🌍 AI Γεωλόγος Pro: 3D Visualization & Analytics")

# Sidebar για ρυθμίσεις
st.sidebar.header("⚙️ Ρυθμίσεις Προβολής")

uploaded_file = st.sidebar.file_uploader("Ανέβασε Σεισμικό Patch (.npy)", type=["npy"])

if uploaded_file is not None:
    # Φόρτωση του αρχικού σήματος για το Overlay
    original_patch = np.load(io.BytesIO(uploaded_file.getvalue()))
    
    if st.sidebar.button("🚀 Εκτέλεση AI Ανάλυσης"):
        with st.spinner("Υπολογισμός..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/octet-stream")}
            response = requests.post("http://127.0.0.1:8000/predict", files=files)
            
            if response.status_code == 200:
                st.session_state['prediction'] = np.load(io.BytesIO(response.content))
                st.session_state['original'] = original_patch
                st.success("Ανάλυση Ολοκληρώθηκε!")

    # Έλεγχος αν υπάρχουν αποτελέσματα στη μνήμη
    if 'prediction' in st.session_state:
        pred = st.session_state['prediction']
        orig = st.session_state['original']
        
        # --- SIDEBAR CONTROLS ---
        st.sidebar.markdown("---")
        axis = st.sidebar.selectbox("Άξονας Τομής", ["Βάθος (Z)", "Πλάτος (Y)", "Μήκος (X)"])
        
        # Δυναμικό Slider ανάλογα με τον άξονα
        max_val = pred.shape[0]-1 if axis == "Βάθος (Z)" else (pred.shape[1]-1 if axis == "Πλάτος (Y)" else pred.shape[2]-1)
        slice_idx = st.sidebar.slider("Επιλογή Φέτας", 0, max_val, max_val//2)
        
        opacity = st.sidebar.slider("Διαφάνεια AI (Overlay)", 0.0, 1.0, 0.7)

        # --- ΚΥΡΙΩΣ ΠΑΝΕΛ ---
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"Ανάλυση Τομής: {axis} στα {slice_idx}")
            
            # Εξαγωγή της σωστής φέτας
            if axis == "Βάθος (Z)":
                s_orig = orig[slice_idx, :, :]
                s_pred = pred[slice_idx, :, :]
            elif axis == "Πλάτος (Y)":
                s_orig = orig[:, slice_idx, :]
                s_pred = pred[:, slice_idx, :]
            else:
                s_orig = orig[:, :, slice_idx]
                s_pred = pred[:, :, slice_idx]

            # Δημιουργία του Overlay Plot με Matplotlib
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(s_orig, cmap='gray') # Το αρχικό σήμα σε γκρι
            img = ax.imshow(s_pred, cmap='jet', alpha=opacity) # Το AI πάνω από το γκρι
            plt.colorbar(img, ax=ax, label="Rock Type Index")
            ax.axis('off')
            st.pyplot(fig)

        with col2:
            st.subheader("📊 Στατιστικά Όγκου")
            
            # Υπολογισμός κατανομής πετρωμάτων
            unique, counts = np.unique(pred, return_counts=True)
            counts_dict = dict(zip(unique, counts))
            
            data = []
            for idx, name in CLASS_NAMES.items():
                if idx in counts_dict:
                    data.append({"Πέτρωμα": name, "Ποσοστό %": (counts_dict[idx] / pred.size) * 100})
            
            df = pd.DataFrame(data)
            
            # Pie Chart με Plotly
            fig_pie = px.pie(df, values='Ποσοστό %', names='Πέτρωμα', color_discrete_sequence=px.colors.sequential.RdBu)
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.table(df)
            
            
            
        # --- 3D INTERACTIVE CUBE (SOTA MULTI-LAYER MINIATURE) ---
        st.markdown("---")
        st.subheader("🧊 Διαδραστική 3D Μικρογραφία (Πολλαπλή Απομόνωση)")
        st.write("Επίλεξε ένα ή περισσότερα πετρώματα για ταυτόχρονη τρισδιάστατη προβολή. Ρύθμισε τις διαιρέσεις για βέλτιστη ταχύτητα.")

        
        # 1. Πολλαπλή επιλογή (Multi-select) αντί για απλό selectbox
        selected_rocks = st.multiselect(
            "Επιλογή Πετρωμάτων για Προβολή:", 
            list(CLASS_NAMES.values()), 
            default=[CLASS_NAMES[2], CLASS_NAMES[3]] # Προεπιλογή: Salt και Sandstone
        )
        
        # 2. Slider για τις διαιρέσεις / βήμα δείγματοληψίας (Downsampling)
        step = st.slider("Επίπεδο Διαιρέσεων (Υψηλότερο = Πιο Ελαφριά Μικρογραφία)", min_value=1, max_value=4, value=2)

        if st.button("✨ Ενημέρωση 3D Μοντέλου"):
            if not selected_rocks:
                st.warning("⚠️ Παρακαλώ επίλεξε τουλάχιστον ένα πέτρωμα για προβολή.")
            else:
                with st.spinner("Δημιουργία 3D μικρογραφίας χώρου..."):
                    
                    fig_3d = go.Figure()
                    
                    # 3. Σχεδίαση κάθε επιλεγμένου στρώματος με το δικό του χρώμα και όνομα στη λεζάντα
                    for rock_name in selected_rocks:
                        # Εύρεση του ID της κλάσης
                        target_idx = [k for k, v in CLASS_NAMES.items() if v == rock_name][0]
                        
                        # Εφαρμογή των διαιρέσεων (step) στους άξονες του κύβου
                        z_coords, y_coords, x_coords = np.where(pred[::step, ::step, ::step] == target_idx)
                        
                        if len(x_coords) > 0:
                            fig_3d.add_trace(go.Scatter3d(
                                x=x_coords * step,
                                y=y_coords * step,
                                z=z_coords * step,
                                mode='markers',
                                name=rock_name, # Εμφάνιση ονόματος στη λεζάντα
                                marker=dict(
                                    size=3,
                                    opacity=0.5 # Διαφάνεια για να βλέπουμε μέσα από τα στρώματα
                                )
                            ))
                    
                    # Ρυθμίσεις του 3D layout για ελεύθερη περιστροφή
                    fig_3d.update_layout(
                        margin=dict(l=0, r=0, b=0, t=0),
                        scene=dict(
                            xaxis_title='Μήκος (X)',
                            yaxis_title='Πλάτος (Y)',
                            zaxis_title='Βάθος (Z)',
                            aspectmode='cube'
                        ),
                        legend=dict(
                            title="Γεωλογικά Στρώματα",
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    st.plotly_chart(fig_3d, use_container_width=True)

else:
    st.warning("👈 Ανέβασε ένα αρχείο .npy από το sidebar για να ξεκινήσουμε.")

                
                
                
                
                
                