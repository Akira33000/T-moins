import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cdist
import io

# Configuration de la page
st.set_page_config(
    page_title="Sélection de communes témoins",
    page_icon="🏙️",
    layout="wide"
)

# Titre de l'application
st.title("Sélection de communes témoins")

# Sidebar pour les contrôles
with st.sidebar:
    st.header("Configuration")
    
    # Chargement des fichiers
    file_cible = st.file_uploader("Fichier des communes cibles (CSV/XLSX)", type=["csv", "xlsx"])
    file_communes = st.file_uploader("Base des communes (CSV/XLSX)", type=["csv", "xlsx"])
    
    # Champ de jointure
    champ = st.text_input("Champ de jointure", value="code_insee")
    
    # Configuration du clustering
    st.subheader("Paramètres de clustering")
    
    # Option pour déterminer automatiquement le nombre de clusters
    auto_k = st.checkbox("Déterminer automatiquement le nombre de clusters", value=False)
    
    # Si non automatique, demander le nombre de clusters
    if not auto_k:
        nb_clusters = st.number_input("Nombre de clusters", min_value=2, max_value=10, value=5)
    
    # Nombre de communes témoins
    nombre_temoin = 1
    
    # Bouton d'exécution
    run_button = st.button("Exécuter l'analyse")

# Fonctions utilitaires
def normalize_code_insee(code):
    """Normalise un code INSEE pour assurer qu'il a 5 caractères"""
    if pd.isna(code):
        return None
    
    # Convertir en chaîne
    code = str(code).strip()
    
    # Si c'est un nombre à 4 chiffres, ajouter un zéro devant
    if code.isdigit() and len(code) == 4:
        return '0' + code
    
    # Si c'est un nombre à 5 chiffres ou plus, ou si ce n'est pas un nombre, retourner tel quel
    return code

def load_data(file):
    """Charge les données à partir d'un fichier CSV ou Excel"""
    if file is None:
        return None
    
    try:
        if file.name.endswith('.csv'):
            try:
                data = pd.read_csv(file, sep=',')
            except:
                data = pd.read_csv(file, sep=';')
        elif file.name.endswith('.xlsx'):
            data = pd.read_excel(file)
        else:
            return None
        
        # Nettoyer les noms de colonnes
        data.columns = [col.strip().lower().replace(' ', '_') for col in data.columns]
        
        return data
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {e}")
        return None

def normalize_data(data, quantitatifs, qualitatifs):
    """Normalise les données pour le clustering"""
    # Créer une copie pour éviter de modifier les données originales
    normalized = data.copy()
    
    # Traiter les valeurs manquantes
    for col in quantitatifs:
        if col in normalized.columns:
            normalized[col] = normalized[col].fillna(normalized[col].median())
    
    # Normalisation des variables quantitatives
    if quantitatifs:
        scaler = StandardScaler()
        normalized[quantitatifs] = scaler.fit_transform(normalized[quantitatifs])
    
    # Encodage one-hot des variables qualitatives
    for col in qualitatifs:
        if col in normalized.columns:
            normalized[col] = normalized[col].fillna("")
            dummies = pd.get_dummies(normalized[col], prefix=col)
            normalized = pd.concat([normalized, dummies], axis=1)
            normalized = normalized.drop(col, axis=1)
    
    return normalized

def perform_clustering(data, n_clusters):
    """Effectue le clustering des données"""
    # Sélectionner les colonnes numériques
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    
    # Appliquer K-means
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(numeric_data)
    
    return model.labels_, model.cluster_centers_

def determine_optimal_k(data, max_k=10):
    """Détermine le nombre optimal de clusters"""
    # Sélectionner les colonnes numériques
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    
    # Calculer l'inertie pour différentes valeurs de k
    inertias = []
    k_values = range(1, min(max_k + 1, len(numeric_data) // 5 + 1))
    
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(numeric_data)
        inertias.append(model.inertia_)
    
    # Trouver le coude
    if len(k_values) > 2:
        diffs = np.diff(inertias)
        k_optimal = np.argmax(np.diff(diffs)) + 2
    else:
        k_optimal = 2
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, inertias, 'o-')
    ax.axvline(x=k_optimal, color='red', linestyle='--')
    ax.set_xlabel('Nombre de clusters')
    ax.set_ylabel('Inertie')
    ax.set_title('Méthode du coude')
    
    return k_optimal, fig


def select_communes_temoins(communes_data, communes_cibles, clusters, nb_temoin_par_cible):
    """Sélectionne les communes témoins pour chaque commune cible"""
    # Convertir nb_temoin_par_cible en entier
    nb_temoin = int(nb_temoin_par_cible)
    
    # Créer un DataFrame avec les clusters
    communes_with_clusters = communes_data.copy()
    communes_with_clusters['cluster'] = clusters
    
    # Identifier les clusters des communes cibles
    communes_cibles_with_clusters = communes_with_clusters.loc[communes_with_clusters.index.isin(communes_cibles.index)]
    
    # Initialiser le DataFrame des communes témoins
    communes_temoins = pd.DataFrame()
    
    # Pour chaque commune cible
    for idx, commune_cible in communes_cibles_with_clusters.iterrows():
        # Trouver le cluster de la commune cible
        cluster_cible = commune_cible['cluster']
        
        # Sélectionner les communes du même cluster (sauf la commune cible elle-même)
        communes_meme_cluster = communes_with_clusters[
            (communes_with_clusters['cluster'] == cluster_cible) & 
            (~communes_with_clusters.index.isin([idx]))
        ]
        
        # Si pas assez de communes dans le cluster, prendre ce qu'il y a
        nb_temoin_effectif = min(nb_temoin, len(communes_meme_cluster))
        
        if nb_temoin_effectif > 0:
            # Sélectionner aléatoirement les communes témoins
            temoins_pour_cible = communes_meme_cluster.sample(nb_temoin_effectif, random_state=42)
            
            # Ajouter une colonne indiquant la commune cible correspondante
            temoins_pour_cible['commune_cible'] = idx
            
            # Ajouter au DataFrame des communes témoins
            communes_temoins = pd.concat([communes_temoins, temoins_pour_cible])
    
    return communes_temoins

def format_results_1to1(communes_temoins):
    """
    Formatage 1 pour 1 : chaque commune cible n'est associée qu'à un seul témoin,
    uniquement avec les colonnes code_insee_cible et code_insee_temoin, sans doublons.
    """
    # On garde le premier témoin pour chaque cible
    resultats = (
        communes_temoins.reset_index()
        .drop_duplicates(subset=['commune_cible'])  # 1 témoin par cible
        .rename(columns={communes_temoins.index.name: 'code_insee_temoin'})
        [['commune_cible', 'code_insee_temoin']]
        .rename(columns={'commune_cible': 'code_insee_cible'})
        .reset_index(drop=True)
    )
    return resultats


def export_to_excel(communes_cibles, resultats_formated):
    """Exporte les résultats en Excel"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        communes_cibles.reset_index().to_excel(writer, sheet_name='Communes cibles', index=False)
        resultats_formated.to_excel(writer, sheet_name='Communes témoins', index=False)
    
    output.seek(0)
    return output.getvalue()

# Corps principal de l'application
if file_cible is not None and file_communes is not None:
    # Chargement des données
    with st.spinner("Chargement des données..."):
        cible_data = load_data(file_cible)
        communes_data = load_data(file_communes)
        
        if cible_data is not None and communes_data is not None:
            st.success("Données chargées avec succès!")
            
            # Vérifier que le champ de jointure est présent
            if champ not in cible_data.columns or champ not in communes_data.columns:
                st.error(f"Le champ '{champ}' n'est pas présent dans les deux fichiers.")
                st.stop()
            
            # Normaliser les codes INSEE
            cible_data[champ] = cible_data[champ].apply(normalize_code_insee)
            communes_data[champ] = communes_data[champ].apply(normalize_code_insee)
            
            # Définir le champ comme index
            if cible_data.index.name != champ:
                cible_data = cible_data.set_index(champ)
            
            if communes_data.index.name != champ:
                communes_data = communes_data.set_index(champ)
            
            # Affichage des aperçus
            st.subheader("Aperçu des communes cibles")
            st.dataframe(cible_data.reset_index().head())
            
            st.subheader("Aperçu de la base des communes")
            st.dataframe(communes_data.reset_index().head())
            
            # Sélection des variables
            st.subheader("Sélection des variables pour le clustering")
            
            # Liste des colonnes numériques et catégorielles
            numeric_cols = communes_data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = communes_data.select_dtypes(include=['object', 'category']).columns.tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("Variables quantitatives (2-3 recommandées)")
                quantitatifs = st.multiselect(
                    "Sélectionnez les variables quantitatives", 
                    numeric_cols, 
                    default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols[:1],
                    key="quanti_select"
                )
            
            with col2:
                st.write("Variables qualitatives (1-2 recommandées)")
                qualitatifs = st.multiselect(
                    "Sélectionnez les variables qualitatives", 
                    categorical_cols, 
                    default=categorical_cols[:1] if categorical_cols else [],
                    key="quali_select"
                )
            
            # Vérifier qu'au moins une variable est sélectionnée
            if not quantitatifs and not qualitatifs:
                st.warning("Veuillez sélectionner au moins une variable pour le clustering.")
            
            # Exécution de l'analyse
            if run_button:
                with st.spinner("Exécution de l'analyse..."):
                    # Limiter le nombre de variables pour éviter les problèmes
                    if len(quantitatifs) > 3:
                        st.warning("Limitation à 3 variables quantitatives pour assurer la stabilité.")
                        quantitatifs = quantitatifs[:3]
                    
                    if len(qualitatifs) > 2:
                        st.warning("Limitation à 2 variables qualitatives pour assurer la stabilité.")
                        qualitatifs = qualitatifs[:2]
                    
                    # Normalisation des données
                    communes_norm = normalize_data(communes_data, quantitatifs, qualitatifs)
                    
                    # Détermination du nombre optimal de clusters si demandé
                    if auto_k:
                        k_optimal, fig_elbow = determine_optimal_k(communes_norm, max_k=10)
                        nb_clusters = k_optimal
                        st.subheader(f"Nombre optimal de clusters: {k_optimal}")
                        st.pyplot(fig_elbow)
                    
                    # Clustering
                    clusters, centers = perform_clustering(communes_norm, nb_clusters)
                    
                    # Calcul du score de silhouette
                    try:
                        numeric_data = communes_norm.select_dtypes(include=['int64', 'float64'])
                        silhouette = silhouette_score(numeric_data, clusters)
                        st.subheader("Qualité du clustering")
                        st.metric("Score de silhouette", f"{silhouette:.3f}")
                        st.caption("Plus le score est proche de 1, meilleur est le clustering")
                    except Exception as e:
                        st.warning(f"Impossible de calculer le score de silhouette: {e}")
                    
                    # Sélection des communes témoins
                    communes_temoins = select_communes_temoins(communes_data, cible_data, clusters, nombre_temoin)

                    resultats_formated = format_results_1to1(communes_temoins)

                    
                    # Affichage des résultats
                    st.subheader("Résultats")
                    st.write(f"Nombre de communes cibles: {len(cible_data)}")
                    st.write(f"Nombre de communes témoins sélectionnées: {len(communes_temoins)}")
                    
                    # Afficher les communes témoins alignées avec les cibles
                    st.subheader("Communes témoins et leurs cibles")
                    st.dataframe(resultats_formated)
                    
                    # Export des résultats
                    excel_data = export_to_excel(cible_data, resultats_formated)
                    st.download_button(
                        label="Télécharger les résultats (Excel)",
                        data=excel_data,
                        file_name="resultats_communes_temoins.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
else:
    st.info("Veuillez télécharger les fichiers de données pour commencer l'analyse.")
