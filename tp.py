

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import pi
import warnings
import os
import time

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE

import hdbscan
import umap

warnings.filterwarnings('ignore')

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")
sns.set_context("notebook", font_scale=1.1)

# Creation des dossiers
os.makedirs('figures', exist_ok=True)



# ==================================================================================================
# 1. CHARGEMENT DES DONNEES
# ==================================================================================================


df = pd.read_csv('data/CC GENERAL.csv')
print(f"Dataset charge: {df.shape[0]} lignes, {df.shape[1]} colonnes")
print(f"Colonnes: {df.columns.tolist()}")

cust_ids = df['CUST_ID']
df_processed = df.drop('CUST_ID', axis=1)
feature_names = df_processed.columns.tolist()
print(f"Variables analysees: {len(feature_names)}")

# ==================================================================================================
# 2. ANALYSE EXPLORATOIRE - STATISTIQUES
# ==================================================================================================


stats_df = df_processed.describe().T
stats_df['skewness'] = df_processed.skew()
stats_df['missing'] = df_processed.isnull().sum()
stats_df['missing_pct'] = (df_processed.isnull().sum() / len(df_processed)) * 100
stats_df.to_csv('statistiques_descriptives.csv')
print("Statistiques sauvegardees dans 'statistiques_descriptives.csv'")
print(stats_df[['mean', 'std', 'min', 'max', 'skewness', 'missing']].round(2).head(10))

# ==================================================================================================
# 3. FIGURE 1: HISTOGRAMMES
# ==================================================================================================


n_vars = len(feature_names)
n_cols = 4
n_rows = (n_vars + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten()

for i, col in enumerate(feature_names):
    axes[i].hist(df_processed[col].dropna(), bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].set_title(f'{col}\n(skewness={df_processed[col].skew():.2f})', fontsize=10)
    axes[i].set_xlabel(col, fontsize=8)
    axes[i].set_ylabel('Frequence', fontsize=8)
    axes[i].axvline(df_processed[col].mean(), color='red', linestyle='--', linewidth=1.5, label='Moyenne')
    axes[i].axvline(df_processed[col].median(), color='green', linestyle='--', linewidth=1.5, label='Mediane')
    if i < 4:
        axes[i].legend(fontsize=8)

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Figure 1: Distribution de toutes les variables', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/01_histograms.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 1: 01_histograms.png")

# ==================================================================================================
# 4. FIGURE 2: BOXPLOTS
# ==================================================================================================
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
axes = axes.flatten()

for i, col in enumerate(feature_names):
    data = df_processed[col].dropna()
    bp = axes[i].boxplot(data, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    axes[i].set_title(col, fontsize=10)
    q1, q3 = data.quantile(0.25), data.quantile(0.75)
    iqr = q3 - q1
    n_outliers = sum((data < q1 - 1.5*iqr) | (data > q3 + 1.5*iqr))
    axes[i].text(0.7, 0.95, f'Outliers: {n_outliers}', transform=axes[i].transAxes, fontsize=8)

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Figure 2: Boxplots pour detection des outliers', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/02_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 2: 02_boxplots.png")

# ==================================================================================================
# 5. FIGURE 3: QQ-PLOTS
# ==================================================================================================
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(feature_names):
    stats.probplot(df_processed[col].dropna(), dist="norm", plot=axes[i])
    axes[i].set_title(col, fontsize=9)

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.suptitle('Figure 3: QQ-Plots pour verifier la normalite', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/03_qqplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 3: 03_qqplots.png")

# ==================================================================================================
# 6. FIGURE 4: MATRICE DE CORRELATION
# ==================================================================================================
corr_matrix = df_processed.corr()

plt.figure(figsize=(20, 18))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, annot_kws={'size': 7})
plt.title('Figure 4: Matrice de correlation des 18 variables', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig('figures/04_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 4: 04_correlation_matrix.png")

# ==================================================================================================
# 7. FIGURE 5: HEATMAP HAUTES CORRELATIONS
# ==================================================================================================
plt.figure(figsize=(14, 12))
high_corr = corr_matrix[abs(corr_matrix) > 0.5]
sns.heatmap(high_corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, square=True)
plt.title('Figure 5: Correlations fortes (|r| > 0.5)', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/05_high_correlations.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 5: 05_high_correlations.png")

# ==================================================================================================
# 8. FIGURE 6: SCATTER PLOTS
# ==================================================================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()
pairs = [('PURCHASES', 'PAYMENTS'), ('PURCHASES', 'CREDIT_LIMIT'),
         ('CASH_ADVANCE', 'CASH_ADVANCE_FREQUENCY'), ('BALANCE', 'PURCHASES'),
         ('PURCHASES_TRX', 'PURCHASES'), ('ONEOFF_PURCHASES', 'PURCHASES')]

for idx, (var1, var2) in enumerate(pairs):
    axes[idx].scatter(df_processed[var1], df_processed[var2], alpha=0.3, s=10, c='steelblue')
    axes[idx].set_xlabel(var1)
    axes[idx].set_ylabel(var2)
    axes[idx].set_title(f'{var1} vs {var2}\ncorr = {corr_matrix.loc[var1, var2]:.3f}')
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Figure 6: Scatter plots des variables fortement correlees', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/06_scatter_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 6: 06_scatter_plots.png")

# ==================================================================================================
# 9. PRETRAITEMENT
# ==================================================================================================


# Imputation KNN
print("\n--- Imputation KNN ---")
print(f"Valeurs manquantes avant: {df_processed.isnull().sum().sum()}")
knn_imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(knn_imputer.fit_transform(df_processed), columns=df_processed.columns)
print(f"Valeurs manquantes apres: {df_imputed.isnull().sum().sum()}")

# Standardisation
print("\n--- Standardisation ---")
scaler = StandardScaler()
X = scaler.fit_transform(df_imputed)
print(f"Donnees standardisees: forme {X.shape}")

# Detection outliers Isolation Forest
print("\n--- Detection Outliers ---")
iso_forest = IsolationForest(contamination=0.1, random_state=42)
outliers_if = iso_forest.fit_predict(X)
n_outliers = sum(outliers_if == -1)
print(f"Outliers detectes: {n_outliers} ({n_outliers/len(X)*100:.1f}%)")

df_clean = df_imputed.copy()
df_clean['is_outlier'] = outliers_if

# ==================================================================================================
# 10. PCA ANALYSIS
# ==================================================================================================

pca = PCA()
X_pca_full = pca.fit_transform(X)

# Figure 7: Variance expliquee
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(1, len(pca.explained_variance_ratio_)+1),
            pca.explained_variance_ratio_, alpha=0.7, color='steelblue')
axes[0].set_xlabel('Composante principale')
axes[0].set_ylabel('Variance expliquee')
axes[0].set_title('Figure 7a: Variance expliquee par composante')
axes[0].grid(True, alpha=0.3)

cumsum = np.cumsum(pca.explained_variance_ratio_)
axes[1].plot(range(1, len(cumsum)+1), cumsum, 'bo-', linewidth=2, markersize=8)
axes[1].axhline(y=0.90, color='r', linestyle='--', label='90%')
axes[1].axhline(y=0.95, color='g', linestyle='--', label='95%')
axes[1].set_xlabel('Nombre de composantes')
axes[1].set_ylabel('Variance cumulee')
axes[1].set_title('Figure 7b: Variance cumulee')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/07_pca_variance.png', dpi=150, bbox_inches='tight')
plt.close()

# Figure 8: Cercle de correlation
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X)
loadings = pca_2d.components_.T

plt.figure(figsize=(14, 12))
circle = plt.Circle((0, 0), 1, fill=False, linestyle='--', color='gray', alpha=0.5)
plt.gca().add_artist(circle)

for i, var in enumerate(feature_names):
    plt.arrow(0, 0, loadings[i, 0]*0.8, loadings[i, 1]*0.8,
              head_width=0.05, head_length=0.05, fc='red', ec='red', alpha=0.7)
    plt.text(loadings[i, 0]*0.85, loadings[i, 1]*0.85, var, fontsize=9, fontweight='bold')

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.2)
plt.title('Figure 8: Cercle de correlation - Contribution des variables', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.savefig('figures/08_pca_loadings.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figures 7 et 8: pca_variance.png, pca_loadings.png")

# ==================================================================================================
# 11. OPTIMISATION DU NOMBRE DE CLUSTERS
# ==================================================================================================


K_range = range(2, 11)
inertias = []
silhouette_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, labels))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].axvline(x=4, color='r', linestyle='--', linewidth=2, label='k optimal (4)')
axes[0].set_xlabel('Nombre de clusters (k)')
axes[0].set_ylabel('Inertie')
axes[0].set_title('Figure 9a: Methode du Coude (Elbow)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
axes[1].axvline(x=4, color='b', linestyle='--', linewidth=2, label='k optimal (4)')
axes[1].set_xlabel('Nombre de clusters (k)')
axes[1].set_ylabel('Score Silhouette')
axes[1].set_title('Figure 9b: Score Silhouette')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Figure 9: Optimisation du nombre de clusters', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/09_cluster_optimization.png', dpi=150, bbox_inches='tight')
plt.close()

optimal_k = K_range[np.argmax(silhouette_scores)]
print(f"Nombre optimal de clusters: k = {optimal_k}")
print("Figure 9: 09_cluster_optimization.png")

# ==================================================================================================
# 12. SILHOUETTE PLOT
# ==================================================================================================
kmeans_opt = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_opt = kmeans_opt.fit_predict(X)
silhouette_vals = silhouette_samples(X, labels_opt)

fig, ax = plt.subplots(figsize=(12, 8))
y_lower = 10

for i in range(optimal_k):
    ith_cluster_silhouette = silhouette_vals[labels_opt == i]
    ith_cluster_silhouette.sort()
    size_cluster = ith_cluster_silhouette.shape[0]
    y_upper = y_lower + size_cluster
    color = plt.cm.tab10(i / optimal_k)
    ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette,
                     facecolor=color, edgecolor=color, alpha=0.7)
    ax.text(-0.05, y_lower + size_cluster/2, f'Cluster {i}', fontsize=10)
    y_lower = y_upper + 10

ax.axvline(x=silhouette_scores[optimal_k-2], color='red', linestyle='--',
           label=f'Silhouette moyenne = {silhouette_scores[optimal_k-2]:.3f}')
ax.set_xlabel('Coefficient de silhouette')
ax.set_ylabel('Cluster')
ax.set_title(f'Figure 10: Plot des silhouettes pour k={optimal_k}', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('figures/10_silhouette_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 10: 10_silhouette_plot.png")

# ==================================================================================================
# 13. CLUSTERING AVEC MULTIPLES ALGORITHMES
# ==================================================================================================


results = {}

print("\n--- K-Means ---")
results['K-Means'] = kmeans_opt.fit_predict(X)

print("\n--- DBSCAN ---")
dbscan = DBSCAN(eps=2.5, min_samples=5)
results['DBSCAN'] = dbscan.fit_predict(X)
print(f"  Clusters: {len(set(results['DBSCAN'])) - (1 if -1 in results['DBSCAN'] else 0)}")
print(f"  Bruit: {sum(results['DBSCAN'] == -1)} ({sum(results['DBSCAN'] == -1)/len(X)*100:.1f}%)")

print("\n--- GMM ---")
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
results['GMM'] = gmm.fit_predict(X)

print("\n--- Agglomerative ---")
agglo = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
results['Agglomerative'] = agglo.fit_predict(X)

print("\n--- Spectral ---")
spectral = SpectralClustering(n_clusters=optimal_k, random_state=42, affinity='nearest_neighbors')
results['Spectral'] = spectral.fit_predict(X)

print("\n--- HDBSCAN ---")
hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
results['HDBSCAN'] = hdb.fit_predict(X)
print(f"  Clusters: {len(set(results['HDBSCAN'])) - (1 if -1 in results['HDBSCAN'] else 0)}")
print(f"  Bruit: {sum(results['HDBSCAN'] == -1)} ({sum(results['HDBSCAN'] == -1)/len(X)*100:.1f}%)")

# ==================================================================================================
# 14. EVALUATION COMPARATIVE
# ==================================================================================================


evaluation = []
for name, labels in results.items():
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters >= 2:
        sil = silhouette_score(X, labels)
        db = davies_bouldin_score(X, labels)
        ch = calinski_harabasz_score(X, labels)
    else:
        sil, db, ch = np.nan, np.nan, np.nan

    evaluation.append({
        'Algorithme': name,
        'Clusters': n_clusters,
        'Bruit': sum(labels == -1) if -1 in labels else 0,
        'Silhouette': round(sil, 4) if not np.isnan(sil) else None,
        'Davies-Bouldin': round(db, 4) if not np.isnan(db) else None,
        'Calinski-Harabasz': round(ch, 2) if not np.isnan(ch) else None
    })

df_eval = pd.DataFrame(evaluation).sort_values('Silhouette', ascending=False)
print("\n" + "="*80)
print(df_eval.to_string(index=False))
df_eval.to_csv('comparatif_performances.csv', index=False)
print("\nTableau sauvegarde dans 'comparatif_performances.csv'")

# ==================================================================================================
# 15. FIGURE 11: BARPLOT COMPARATIF
# ==================================================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Silhouette
bars1 = axes[0].bar(df_eval['Algorithme'], df_eval['Silhouette'], color='steelblue', alpha=0.7)
axes[0].set_ylabel('Score Silhouette')
axes[0].set_title('Figure 11a: Comparaison des scores Silhouette')
axes[0].set_ylim(0, 0.6)
axes[0].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, df_eval['Silhouette']):
    if val:
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)

# Davies-Bouldin
bars2 = axes[1].bar(df_eval['Algorithme'], df_eval['Davies-Bouldin'], color='coral', alpha=0.7)
axes[1].set_ylabel('Indice Davies-Bouldin (plus bas = meilleur)')
axes[1].set_title('Figure 11b: Comparaison des indices Davies-Bouldin')
axes[1].grid(True, alpha=0.3, axis='y')

plt.xticks(rotation=45, ha='right')
plt.suptitle('Figure 11: Comparaison des performances des algorithmes', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/11_performance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 11: 11_performance_comparison.png")

# ==================================================================================================
# 16. CENTROIDS HEATMAP
# ==================================================================================================
centroids = kmeans_opt.cluster_centers_
centroids_df = pd.DataFrame(centroids, columns=feature_names)
centroids_df.index = [f'Cluster_{i}' for i in range(optimal_k)]

plt.figure(figsize=(16, 8))
sns.heatmap(centroids_df.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            cbar_kws={'label': 'Ecart-type (Z-score)'})
plt.title('Figure 12: Carte thermique des centroides K-Means', fontsize=14, fontweight='bold')
plt.xlabel('Cluster')
plt.ylabel('Variable')
plt.tight_layout()
plt.savefig('figures/12_centroids_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 12: 12_centroids_heatmap.png")

# ==================================================================================================
# 17. ANALYSE STABILITE
# ==================================================================================================


from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari_scores = []
nmi_scores = []

for i in range(10):
    kmeans_test = KMeans(n_clusters=optimal_k, random_state=i, n_init=1)
    labels_test = kmeans_test.fit_predict(X)
    if i == 0:
        labels_ref = labels_test
    else:
        ari_scores.append(adjusted_rand_score(labels_ref, labels_test))
        nmi_scores.append(normalized_mutual_info_score(labels_ref, labels_test))

print(f"ARI moyen: {np.mean(ari_scores):.4f} (+/- {np.std(ari_scores):.4f})")
print(f"NMI moyen: {np.mean(nmi_scores):.4f} (+/- {np.std(nmi_scores):.4f})")

# ==================================================================================================
# 18. RADAR CHART
# ==================================================================================================


# Variables pour le radar
radar_vars = ['PURCHASES', 'CASH_ADVANCE', 'BALANCE', 'PAYMENTS',
              'CREDIT_LIMIT', 'PURCHASES_FREQUENCY', 'CASH_ADVANCE_FREQUENCY', 'ONEOFF_PURCHASES']

# Filtrer les variables existantes
radar_vars = [v for v in radar_vars if v in centroids_df.columns]

radar_data = centroids_df[radar_vars].values
N = len(radar_vars)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
cluster_names = ['Cluster 0: Dormants', 'Cluster 1: Moderés', 'Cluster 2: Premium', 'Cluster 3: Risque']

for i in range(optimal_k):
    values = radar_data[i].tolist()
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=cluster_names[i])
    ax.fill(angles, values, alpha=0.15, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_vars, fontsize=9, rotation=45, ha='right')
ax.set_ylim(-1.5, 2.0)
ax.set_yticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0])
ax.set_yticklabels(['-1.5', '-1.0', '-0.5', '0', '0.5', '1.0', '1.5', '2.0'], fontsize=8)
ax.set_title('Figure 13: Profils radar des clusters', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig('figures/13_radar_chart.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 13: 13_radar_chart.png")

# ==================================================================================================
# 19. RADAR CHART SIMPLIFIE
# ==================================================================================================
radar_vars_simple = ['PURCHASES', 'CASH_ADVANCE', 'BALANCE', 'PAYMENTS']
radar_vars_simple = [v for v in radar_vars_simple if v in centroids_df.columns]
radar_data_simple = centroids_df[radar_vars_simple].values
N_simple = len(radar_vars_simple)
angles_simple = [n / float(N_simple) * 2 * pi for n in range(N_simple)]
angles_simple += angles_simple[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})

for i in range(optimal_k):
    values = radar_data_simple[i].tolist()
    values += values[:1]
    ax.plot(angles_simple, values, 'o-', linewidth=2, color=colors[i], label=cluster_names[i])
    ax.fill(angles_simple, values, alpha=0.15, color=colors[i])

ax.set_xticks(angles_simple[:-1])
ax.set_xticklabels(radar_vars_simple, fontsize=11)
ax.set_ylim(-1.0, 1.5)
ax.set_title('Figure 14: Profils radar simplifies (4 variables cles)', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig('figures/14_radar_chart_simple.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 14: 14_radar_chart_simple.png")

# ==================================================================================================
# 20. UMAP VISUALIZATION
# ==================================================================================================


umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
X_umap = umap_reducer.fit_transform(X)
print(f"UMAP termine: forme {X_umap.shape}")

# Figure 15: UMAP Outliers analysis
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

sc1 = axes[0].scatter(X_umap[:, 0], X_umap[:, 1], c=outliers_if, cmap='coolwarm', alpha=0.5, s=8)
axes[0].set_title('Figure 15a: UMAP - Points normaux (bleu) vs Outliers (rouge)', fontsize=12)
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')
plt.colorbar(sc1, ax=axes[0])

sc2 = axes[1].scatter(X_umap[:, 0], X_umap[:, 1], c=results['DBSCAN'], cmap='tab10', alpha=0.5, s=8)
axes[1].set_title('Figure 15b: UMAP - Clusters DBSCAN (label -1 = bruit)', fontsize=12)
axes[1].set_xlabel('UMAP 1')
axes[1].set_ylabel('UMAP 2')
plt.colorbar(sc2, ax=axes[1])

plt.suptitle('Figure 15: Analyse des outliers par UMAP', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/15_umap_outliers_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 15: 15_umap_outliers_analysis.png")

# Figure 16: UMAP clusters comparison
models_to_plot = ['K-Means', 'DBSCAN', 'GMM', 'Agglomerative', 'Spectral', 'HDBSCAN']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, model_name in enumerate(models_to_plot):
    if model_name in results:
        labels = results[model_name]
        scatter = axes[idx].scatter(X_umap[:, 0], X_umap[:, 1], c=labels, cmap='tab10', alpha=0.5, s=8)
        axes[idx].set_title(f'{model_name}', fontsize=12)
        axes[idx].set_xlabel('UMAP 1')
        axes[idx].set_ylabel('UMAP 2')
        plt.colorbar(scatter, ax=axes[idx])

plt.suptitle('Figure 16: Comparaison des algorithmes sur l\'espace UMAP', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/16_umap_clusters_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 16: 16_umap_clusters_comparison.png")

# ==================================================================================================
# 21. T-SNE VISUALIZATION
# ==================================================================================================


n_tsne = min(3000, X.shape[0])
np.random.seed(42)
indices_tsne = np.random.choice(X.shape[0], n_tsne, replace=False)
X_tsne_sample = X[indices_tsne]

print(f"Calcul de t-SNE sur {n_tsne} echantillons...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_tsne_sample)
print("t-SNE termine")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, model_name in enumerate(models_to_plot):
    if model_name in results:
        labels = results[model_name][indices_tsne]
        scatter = axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='tab10', alpha=0.5, s=8)
        axes[idx].set_title(f'{model_name} (t-SNE)', fontsize=12)
        axes[idx].set_xlabel('t-SNE 1')
        axes[idx].set_ylabel('t-SNE 2')
        plt.colorbar(scatter, ax=axes[idx])

plt.suptitle('Figure 17: Comparaison des algorithmes sur l\'espace t-SNE', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/17_tsne_comparaison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 17: 17_tsne_comparaison.png")

# ==================================================================================================
# 22. COMPARAISON UMAP vs t-SNE
# ==================================================================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# UMAP
sc1 = axes[0].scatter(X_umap[indices_tsne, 0], X_umap[indices_tsne, 1],
                      c=results['K-Means'][indices_tsne], cmap='tab10', alpha=0.5, s=8)
axes[0].set_title('Figure 18a: UMAP', fontsize=12)
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')
plt.colorbar(sc1, ax=axes[0])

# t-SNE
sc2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                      c=results['K-Means'][indices_tsne], cmap='tab10', alpha=0.5, s=8)
axes[1].set_title('Figure 18b: t-SNE', fontsize=12)
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
plt.colorbar(sc2, ax=axes[1])

plt.suptitle('Figure 18: Comparaison UMAP vs t-SNE pour K-Means', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/18_umap_vs_tsne_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Figure 18: 18_umap_vs_tsne_comparison.png")

# ==================================================================================================
# 23. PROFILS DES CLUSTERS
# ==================================================================================================


for cluster in range(optimal_k):
    size = sum(results['K-Means'] == cluster)
    pct = size/len(X)*100
    print(f"\nCluster {cluster}: {size} clients ({pct:.1f}%)")

    # Top variables positives
    top_pos = centroids_df.iloc[cluster].sort_values(ascending=False).head(5)
    print(f"  Forces: {', '.join([f'{col}({val:.2f})' for col, val in top_pos.items()])}")

    # Top variables negatives
    top_neg = centroids_df.iloc[cluster].sort_values(ascending=True).head(3)
    print(f"  Faiblesses: {', '.join([f'{col}({val:.2f})' for col, val in top_neg.items()])}")

# ==================================================================================================
# 24. RESUME FINAL
# ==================================================================================================


figures_list = [
    "01_histograms.png - Distribution de toutes les variables",
    "02_boxplots.png - Boxplots pour detection des outliers",
    "03_qqplots.png - QQ-plots pour verifier la normalite",
    "04_correlation_matrix.png - Matrice de correlation complete",
    "05_high_correlations.png - Correlations fortes (|r| > 0.5)",
    "06_scatter_plots.png - Scatter plots des variables correlees",
    "07_pca_variance.png - Variance expliquee par PCA",
    "08_pca_loadings.png - Cercle de correlation PCA",
    "09_cluster_optimization.png - Optimisation du nombre de clusters",
    "10_silhouette_plot.png - Plot des silhouettes",
    "11_performance_comparison.png - Comparaison des performances des algorithmes",
    "12_centroids_heatmap.png - Carte thermique des centroides",
    "13_radar_chart.png - Profils radar complets (8 variables)",
    "14_radar_chart_simple.png - Profils radar simplifies (4 variables)",
    "15_umap_outliers_analysis.png - Analyse des outliers par UMAP",
    "16_umap_clusters_comparison.png - Comparaison UMAP des algorithmes",
    "17_tsne_comparaison.png - Comparaison t-SNE des algorithmes",
    "18_umap_vs_tsne_comparison.png - Comparaison UMAP vs t-SNE"
]

for f in figures_list:
    print(f"  {f}")

