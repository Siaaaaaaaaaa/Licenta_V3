import streamlit as st
import joblib
import numpy as np
import pandas as pd

# 1. Configurarea paginii web
st.set_page_config(
    page_title="Movie Success Predictor",
    page_icon="",
    layout="centered"
)

# 2. Încărcarea modelelor salvate anterior
@st.cache_resource # Această linie asigură că modelele se încarcă o singură dată în memorie, nu la fiecare click
def load_models():
    clf = joblib.load('model_clasificare.pkl')
    regressor_log = joblib.load('model_regresie.pkl')
    top_genres = joblib.load('top_genres.pkl')
    return clf, regressor_log, top_genres

try:
    clf, regressor_log, top_genres = load_models()
except FileNotFoundError:
    st.error("❌ Nu s-au găsit fișierele .pkl! Asigură-te că ai rulat scriptul de analiză și că fișierele sunt în același folder cu app.py.")
    st.stop()

# Dicționarul de mapare pentru interpretarea clusterelor
cluster_map = {
    0: "Succes Comercial de Nivel Mediu",
    1: "Producție Independentă și cu Buget Redus",
    2: "Top Blockbuster (Impact Mare pe Piață)"
}

# 3. Interfața Grafică (Design-ul Site-ului)
st.title("🎬 Simulator Economic pentru Industria Cinematografică")
st.markdown("---")
st.write("Introdu datele de producție ale unui film nou pentru a evalua profilul său de succes și încasările estimate.")

# Formular pentru inputurile utilizatorului
with st.form("movie_form"):
    st.subheader("📊 Parametri de Intrare")
    
    # Inputuri numerice
    budget = st.number_input("Buget de Producție (USD):", min_value=1000, value=50000000, step=500000)
    runtime = st.number_input("Durata Filmului (minute):", min_value=1, value=120, step=5)
    
    # Selectarea genurilor cinematografice (Multi-select)
    st.markdown("**Genuri (Alege una sau mai multe opțiuni):**")
    selected_genres = st.multiselect("Selectează genurile care se aplică:", top_genres)
    
    # Butonul de trimitere
    submit_button = st.form_submit_button(label="🔮 Generează Predicția de Succes")

# 4. Logica de Predicție (când utilizatorul apasă pe buton)
if submit_button:
    # Pregătim vectorul de caracteristici (X_new) exact în ordinea din antrenare
    # [budget, runtime, genre_Drama, genre_Comedy...]
    genre_features = [1 if genre in selected_genres else 0 for genre in top_genres]
    X_new = [budget, runtime] + genre_features
    X_new_array = np.array([X_new])
    
    # --- PASUL A: CLASIFICAREA (Tipologia de succes) ---
    # Calculăm probabilitățile pentru fiecare clasă în parte
    probabilitati = clf.predict_proba(X_new_array)[0]
    predicted_cluster = clf.predict(X_new_array)[0]
    tipologie_estimata = cluster_map[predicted_cluster]
    
    # --- PASUL B: REGRESIA LOGARITMATĂ (Încasările în USD) ---
    pred_log_revenue = regressor_log.predict(X_new_array)[0]
    # Aplicăm expm1 pentru a anula transformarea log1p
    pred_real_revenue = np.expm1(pred_log_revenue)
    
    # 5. Afișarea Rezultatelor în Interfață
    st.markdown("---")
    st.header("🎯 Rezultatele Simulării")
    
    # Afișare Categorie Estimator
    if predicted_cluster == 2:
        st.success(f"🏆 **Profil Estimat:** {tipologie_estimata}")
    elif predicted_cluster == 0:
        st.info(f"📈 **Profil Estimat:** {tipologie_estimata}")
    else:
        st.warning(f"🎥 **Profil Estimat:** {tipologie_estimata}")
        
    # Afișare Venituri Estimate din Regresie
    st.metric(
        label="💰 Încasări Estimate (Venit Global)", 
        value=f"${pred_real_revenue:,.2f} USD",
        help="Valoare calculată prin modelul Random Forest Regressor aplicat pe scară logaritmică."
    )
    
    # Afișare Probabilități pe fiecare Cluster (Analiza de Risc)
    st.subheader("📊 Distribuția Probabilităților de Succes")
    
    # Creăm un DataFrame mic pentru un grafic Streamlit curat
    prob_data = pd.DataFrame({
        'Tipologie': [cluster_map[i] for i in sorted(cluster_map.keys())],
        'Probabilitate (%)': [p * 100 for p in probabilitati]
    })
    
    # Desenăm un grafic de bare orizontale direct în Streamlit
    st.bar_chart(data=prob_data, x='Tipologie', y='Probabilitate (%)', color="#2ca02c" if predicted_cluster==2 else "#1f77b4")
    
    # Text explicativ academic pentru licență
    st.caption(
        "Notă academică: Clasificarea folosește un model Random Forest antrenat istoric (acuratețe 95%). "
        "Regresia logistic-transformată capturează tendința generală a încasărilor (R² = 0.43), restul variabilității "
        "fiind atribuită factorilor de artă cinematografică și marketing."
    )