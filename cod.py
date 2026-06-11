import os
import ast
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


base_path = os.path.dirname(__file__)
csv_read = os.path.join(base_path, 'movies_metadata.csv') 
data = pd.read_csv(csv_read)

data.columns = [c.strip().lower().replace(' ', '_') for c in data.columns]
output_file = open("output_data.txt", "w", encoding="utf-8") #scriem datele in fisier pt a fi mai usor de urmarit
output_file.write(f"Afișăm datele inițiale: \n {data.head()}\n\n")

# Curățare minimală pentru grafice
data['budget'] = pd.to_numeric(data['budget'], errors='coerce')
data['revenue'] = pd.to_numeric(data['revenue'], errors='coerce')
data['popularity'] = pd.to_numeric(data['popularity'], errors='coerce')
data['vote_count'] = pd.to_numeric(data['vote_count'], errors='coerce')

# Eliminăm rândurile cu 0 la buget sau venituri pentru a nu distorsiona graficele
df_plot = data[(data['budget'] > 0) & (data['revenue'] > 0)].copy()

output_dir = 'Grafice'
if not os.path.exists(output_dir):
    os.makedirs(output_dir) #salvam graficele intr un folder ptr a fi mai usor de urmarit

#generare grafice inițiale - analiză de date 

#1 distribuție venituri - histogramă 
plt.figure(figsize=(10, 6))
sns.histplot(df_plot['revenue'], bins=30, kde=True, color='blue')
plt.title('Distribuția Veniturilor Filmelor')
plt.xlabel('Venituri ')
plt.ylabel('Frecvență')
plt.savefig(os.path.join(output_dir, '1venituri.png'))
plt.close()

#maj filmelor produc venituri mici - succesul masiv nu e o regulă care reiese din venit 

#relație dintre buget si venit - scatter plot 
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_plot, x='budget', y='revenue', alpha=0.5, color='green')
plt.title('Corelația dintre Buget și Venituri')
plt.xlabel('Buget')
plt.ylabel('Venituri')
plt.savefig(os.path.join(output_dir, '2buget_vs_venituri.png'))
plt.close()
#Bugetul este un predictor puternic pentru venituri, dar nu este singuru - există multe filme cu buget mare care au încasat puțin 

#top 10 genuri de filme - bar chart 
plt.figure(figsize=(12, 6))
df_plot['original_language'].value_counts().head(10).plot(kind='bar', color='orange')
plt.title('Top 10 Limbi Originale (Frecvență)')
plt.xlabel('Limba')
plt.ylabel('Număr de Filme')
plt.savefig(os.path.join(output_dir, '3topfilme.png'))
plt.close()
#multe filme in limba engleza - poate să fie o coloană de care să ne lipsim în modelare 

#matrice de corelație - heatmap 
plt.figure(figsize=(8, 6))
correlation_matrix = df_plot[['budget', 'revenue', 'popularity', 'vote_average', 'vote_count']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice de Corelație a Indicatorilor Cheie')
plt.savefig(os.path.join(output_dir, '4matrice_corelatie.png'))
plt.close()

#Corelația Buget - Revenue (0.73): Este o legătură foarte puternică. Confirmă matematic ce am văzut în graficul verde.
#Corelația Vote Count - Revenue (0.77): Aceasta este cea mai mare surpriză plăcută! Numărul de oameni care votează un film corelează mai bine cu veniturile decât bugetul.
#Corelația Vote Average - Budget (-0.01): Surpriză! Nota filmului nu are nicio legătură cu bugetul. Un film scump nu este neapărat un film mai bun (notat mai bine de public).
#Interpretare generală: Pentru modelele tale de regresie, cele mai importante „unelte” vor fi Bugetul și Numărul de voturi (Popularitatea).

print(f"Cele 4 grafice au fost salvate cu succes în folderul: {output_dir}")

#1. Clusterizare  - încercăm să categorisim profile de bisniz de succes pentru filme - vrem un algoritm care să grupeze filmele care seamănă între ele din punct de vedere al resurselor și al impactului 
#var care definesc succesul unui film sunt bugeul (cât s-a investit), popularitatea, cât a produs și câți oameni l-au validat

features = ['budget','revenue','popularity','vote_count']
X = df_plot[features]

#scalarea datelor 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#metoda elbow - pentru a vedea în câte k - clustere clusterizăm
wcss=[]
for i in range(1,11): 
    kmeans = KMeans (n_clusters=i, init='k-means++',random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

#salvam grafic elbow 
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='b')
plt.title('Metoda Elbow pentru determinarea numărului optim de clustere')
plt.xlabel('Număr de Clustere (k)')
plt.ylabel('WCSS (Inerția)')
plt.xticks(range(1, 11)) # Ne asigurăm că pe axa X apar numerele de la 1 la 10
plt.grid(True)

plt.savefig(os.path.join(output_dir, '6elbow_method.png'))

#aplicam k -means ptr 3 clustere, justificat din graifcu de mai sus
kmeans_final = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
df_plot['cluster'] = kmeans_final.fit_predict(X_scaled)

#vizualizare klustere
stats = df_plot.groupby('cluster')[features].mean()

cluster_map = {
    2:"Top Blockbustere (Impact Mare)",
    0: "Succes Comercial de Nivel Mediu",
    1: "Filme Independente & de Buget Mic", 
}

df_plot['tipologie_film'] = df_plot['cluster'].map(cluster_map)

plt.figure(figsize=(12, 8))
sns.scatterplot(data=df_plot, x='budget', y='revenue', hue='tipologie_film', palette='viridis', s=80)
plt.title('7. Segmentarea Pieței pe Tipologii de Film')
plt.savefig(os.path.join(output_dir, '7vizualizare_tipologii.png'))
plt.close()

avg_stats = df_plot.groupby('tipologie_film')[['budget', 'revenue']].mean().sort_values('revenue')
avg_stats.plot(kind='bar', figsize=(10, 6), color=['#1f77b4', '#ff7f0e'])
plt.title('8. Comparație Buget vs Venit Mediu pe Tipologie')
plt.ylabel('Valoare (USD)')
plt.xticks(rotation=0)
plt.savefig(os.path.join(output_dir, '8comparatie_medii.png'))
plt.close()

output_file.write(f"Mediile pe fiecare cluster:\n{stats}\n\n")
output_file.write(f"Mapare tipologii: {cluster_map}\n")
output_file.close()

print("Etapa de clusterizare finalizată. Graficele noi au fost salvate.")

# clasificare - vrem sa urmarim daca filmele respective pot fi o pierdere sau un castig 

def extract_genres(genre_str):
    try:
        genres = ast.literal_eval(genre_str)
        return [g['name'] for g in genres]
    except:
        return []
    
df_plot['genres_list'] =data['genres'].apply(extract_genres)
top_genres = ['Drama', 'Comedy', 'Thriller', 'Action', 'Romance', 'Adventure', 'Crime', 'Science Fiction', 'Horror', 'Family']
for genre in top_genres:
    df_plot[f'genre_{genre}'] = df_plot['genres_list'].apply(lambda x: 1 if genre in x else 0)


#  Nu includem 'revenue' sau 'vote_count' aici, pentru că acelea sunt rezultate, 
# iar noi vrem să prezicem succesul ÎNAINTE ca filmul să apară.
X = df_plot[['budget', 'runtime', 'popularity'] + [f'genre_{genre}' for genre in top_genres]].fillna(0)
y = df_plot['cluster']    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=[cluster_map[i] for i in sorted(cluster_map.keys())], zero_division=0)

# Scriem rezultatele în fișierul nostru de raport
output_file = open("output_data.txt", "a", encoding="utf-8")
output_file.write(f"\nRezultate etapa clasificare:\n")
output_file.write(f"Acuratețe Generală: {accuracy:.2f}\n")
output_file.write(f"Raport Detaliat:\n{report}\n")
output_file.close()

importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=importances.index, hue=importances.index, palette='viridis', legend=False)
plt.title('Importanța Factorilor în Predicția Tipologiei de Film')
plt.xlabel('Scor de Importanță)')
plt.ylabel('Predictori (Features)')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '9importanta_factori.png'))
plt.close()

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[cluster_map[i] for i in sorted(cluster_map.keys())],
            yticklabels=[cluster_map[i] for i in sorted(cluster_map.keys())])

plt.title('Matricea de Confuzie - Evaluarea Erorilor de Predicție')
plt.xlabel('Predicția Modelului')
plt.ylabel('Realitatea')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '10matrice_confuzie.png'))
plt.close()

print(f"Clasificare finalizată. Acuratețe: {accuracy:.2f}. Graficul 9 & 10 au fost salvate.")

#regresie - prezicerea directa a castigului 
# Adăugăm o constantă mică (+1) pentru a evita log(0), Deoarece log(revenue) poate fi negativ, folosim log1p care e log(x+1)
df_plot['log_revenue'] = np.log1p(df_plot['revenue'])

# datele pentru Regresia Logaritmata
y_log_reg = df_plot['log_revenue'] 
X_log_reg = X 

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log_reg, y_log_reg, test_size=0.2, random_state=42)
regressor_log = RandomForestRegressor(n_estimators=200, random_state=42)
regressor_log.fit(X_train_log, y_train_log)

y_pred_log_scale = regressor_log.predict(X_test_log)

# Inversăm transformarea pentru a reveni la USD reali (EXP)
# Folosim expm1 pentru a anula log1p
y_test_real = np.expm1(y_test_log)
y_pred_real = np.expm1(y_pred_log_scale)

# Evaluare pe CIFRE REALE (USD)
r2 = r2_score(y_test_real, y_pred_real)
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

output_file = open("output_data.txt", "a", encoding="utf-8")
output_file.write(f"\nR-squared (Scor R2): {r2:.2f}\n")
output_file.write(f"Eroarea Medie Absolută: {mae:,.2f} USD\n")
output_file.write(f"Eroarea Patratica: {rmse:,.2f} USD\n")

plt.figure(figsize=(10, 6))
plt.scatter(y_test_real, y_pred_real, alpha=0.4, color='darkgreen')
# Linia de 45 grade
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'k--', lw=2)
plt.title('Venituri Reale vs. Prezise (Scară Log-Transformată)')
plt.xlabel('Venituri Reale (USD)')
plt.ylabel('Venituri Prezise (USD)')
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, '11regresie.png'))
plt.close()

print(f"Etapa de regresie a fost finalzizata")

#salvare modele 
import joblib
joblib.dump(clf, 'model_clasificare.pkl')
joblib.dump(regressor_log, 'model_regresie.pkl')

# Salvăm lista de genuri pentru a ne asigura că aplicația web va folosi exact aceeași ordine a coloanelor 
joblib.dump(top_genres, 'top_genres.pkl')

print("Fișierele .pkl au fost generate cu succes în folderul proiectului!")