# ============================================
# TP Master IA - Classification
# Dataset: Breast Cancer Wisconsin
# ============================================


import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# Maintenant les imports normaux
import numpy as np
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



# Scikit-learn imports
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV, learning_curve, validation_curve
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    log_loss, brier_score_loss, precision_recall_curve, auc
)

# Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

# For advanced bonus
import shap
from sklearn.calibration import calibration_curve

# ------------------------------
# 2. CHARGEMENT ET COMPRÉHENSION DES DONNÉES
# ------------------------------
print("="*60)
print("1. CHARGEMENT ET COMPRÉHENSION DES DONNÉES")
print("="*60)

# Charger le dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(f"\n Shape du dataset: {X.shape}")
print(f" Nombre d'échantillons: {X.shape[0]}")
print(f" Nombre de features: {X.shape[1]}")
print(f" Classes: {data.target_names} (0 = Malin, 1 = Bénin)")

# Statistiques descriptives
print("\nStatistiques descriptives:")
print(X.describe().round(2))

# Vérification des valeurs manquantes
print(f"\n Valeurs manquantes: {X.isnull().sum().sum()}")

# Distribution des classes
print("\n Distribution des classes:")
print(f"Classe 0 (Malin): {(y==0).sum()} échantillons ({((y==0).sum()/len(y))*100:.1f}%)")
print(f"Classe 1 (Bénin): {(y==1).sum()} échantillons ({((y==1).sum()/len(y))*100:.1f}%)")

# ------------------------------
# 3. VISUALISATIONS EXPLORATOIRES
# ------------------------------
print("\n" + "="*60)
print("2. VISUALISATIONS EXPLORATOIRES")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Histogrammes des features
features_to_plot = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean concavity']
for i, feature in enumerate(features_to_plot):
    row, col = i//3, i%3
    ax = axes[row, col]
    ax.hist(X[X.columns[i]][y==0], alpha=0.5, label='Malin', bins=20, color='red')
    ax.hist(X[X.columns[i]][y==1], alpha=0.5, label='Bénin', bins=20, color='green')
    ax.set_title(f'Distribution: {feature}')
    ax.legend()
plt.tight_layout()
plt.savefig('eda_histograms.png', dpi=100, bbox_inches='tight')
plt.show()

# Boxplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, feature in enumerate(features_to_plot):
    row, col = i//3, i%3
    ax = axes[row, col]
    df_box = pd.DataFrame({feature: X[feature], 'Target': y})
    sns.boxplot(x='Target', y=feature, data=df_box, ax=ax, palette=['red', 'green'])
    ax.set_title(f'Boxplot: {feature}')
plt.tight_layout()
plt.savefig('eda_boxplots.png', dpi=100, bbox_inches='tight')
plt.show()

# Matrice de corrélation
plt.figure(figsize=(12, 10))
correlation_matrix = X.corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Matrice de Corrélation des Features')
plt.savefig('eda_correlation_matrix.png', dpi=100, bbox_inches='tight')
plt.show()

print(" Visualisations sauvegardées: 'eda_histograms.png', 'eda_boxplots.png', 'eda_correlation_matrix.png'")

# ------------------------------
# 4. PRÉTRAITEMENT AVANCÉ
# ------------------------------
print("\n" + "="*60)
print("3. PRÉTRAITEMENT AVANCÉ")
print("="*60)

# Split des données 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f" Train set: {X_train.shape[0]} samples")
print(f" Test set: {X_test.shape[0]} samples")

# Standardisation (nécessaire pour KNN, SVM, Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# RobustScaler pour gérer les outliers (Bonus)
robust_scaler = RobustScaler()
X_train_robust = robust_scaler.fit_transform(X_train)
X_test_robust = robust_scaler.transform(X_test)

print(" StandardScaler et RobustScaler appliqués (pour comparaison)")

# ------------------------------
# 5. DÉFINITION DES MODÈLES ET HYPERPARAMÈTRES
# ------------------------------
print("\n" + "="*60)
print("4. DÉFINITION DES MODÈLES")
print("="*60)

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
    'SVM': SVC(probability=True, random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

# Hyperparamètres pour RandomizedSearchCV (Bonus avancé)
param_distributions = {
    'Logistic Regression': {
        'C': np.logspace(-3, 3, 20),
        'penalty': ['l2'],
        'solver': ['lbfgs', 'liblinear']
    },
    'KNN': {
        'n_neighbors': range(3, 31),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    },
    'Decision Tree': {
        'max_depth': range(3, 21),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 11),
        'criterion': ['gini', 'entropy']
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVM': {
        'C': np.logspace(-3, 3, 15),
        'gamma': np.logspace(-3, 3, 15),
        'kernel': ['rbf', 'linear']
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'subsample': [0.6, 0.8, 1.0]
    }
}

# Stratified K-Fold (Bonus)
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ------------------------------
# 6. ENTRAÎNEMENT ET ÉVALUATION COMPLÈTE
# ------------------------------
print("\n" + "="*60)
print("5. ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES")
print("="*60)

results = []

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f" Training: {name}")
    print('='*50)
    
    # Optimisation des hyperparamètres avec RandomizedSearchCV (Bonus avancé)
    print(" Optimisation hyperparamètres (RandomizedSearchCV 5-fold)...")
    
    random_search = RandomizedSearchCV(
        model, 
        param_distributions[name], 
        n_iter=30,  # 30 combinaisons aléatoires
        cv=cv_strategy,
        scoring='f1',
        n_jobs=-1,
        random_state=42,
        verbose=0
    )
    
    random_search.fit(X_train_scaled, y_train)
    best_model = random_search.best_estimator_
    
    print(f" Meilleurs paramètres: {random_search.best_params_}")
    print(f" Meilleur score F1 (CV): {random_search.best_score_:.4f}")
    
    # Prédictions
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # ----- MÉTRIQUES CLASSIQUES -----
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # ----- MÉTRIQUES AVANCÉES (Bonus demandé) -----
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    logloss = log_loss(y_test, y_proba)
    brier = brier_score_loss(y_test, y_proba)
    
    # Validation croisée sur le modèle optimisé (Nested CV - Bonus)
    cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=cv_strategy, scoring='f1')
    
    # Stockage des résultats
    results.append({
        'Modèle': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc,
        'Balanced Accuracy': balanced_acc,
        'MCC': mcc,
        'Cohen Kappa': kappa,
        'Log Loss': logloss,
        'Brier Score': brier,
        'CV F1 Mean': cv_scores.mean(),
        'CV F1 Std': cv_scores.std()
    })
    
    # ----- MATRICE DE CONFUSION -----
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Malin (0)', 'Bénin (1)'],
                yticklabels=['Malin (0)', 'Bénin (1)'])
    plt.title(f'Matrice de Confusion - {name}')
    plt.ylabel('Vérité')
    plt.xlabel('Prédiction')
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print(f"\n Résultats sur Test Set:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    print(f"   ROC AUC: {roc_auc:.4f}")
    print(f"   Balanced Accuracy: {balanced_acc:.4f}")
    print(f"   MCC: {mcc:.4f}")
    print(f"   Cohen's Kappa: {kappa:.4f}")

# ------------------------------
# 7. TABLEAU COMPARATIF DES RÉSULTATS
# ------------------------------
print("\n" + "="*60)
print("6. TABLEAU COMPARATIF DES RÉSULTATS")
print("="*60)

results_df = pd.DataFrame(results)
results_df = results_df.round(4)
print("\n Comparaison des Modèles:")
print(results_df.to_string(index=False))

# Sauvegarde des résultats
results_df.to_csv('model_comparison_results.csv', index=False)
print("\n Résultats sauvegardés: 'model_comparison_results.csv'")

# ------------------------------
# 8. COURBES PRECISION-RECALL (Bonus - plus pertinent pour dataset déséquilibré)
# ------------------------------
print("\n" + "="*60)
print("7. COURBES PRECISION-RECALL (PR AUC)")
print("="*60)

plt.figure(figsize=(10, 8))

for name, model in models.items():
    best_model = models[name]
    # On réentraîne avec les meilleurs paramètres trouvés
    if name in param_distributions:
        rs = RandomizedSearchCV(best_model, param_distributions[name], n_iter=10, cv=cv_strategy, random_state=42)
        rs.fit(X_train_scaled, y_train)
        best_model = rs.best_estimator_
    else:
        best_model.fit(X_train_scaled, y_train)
    
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    plt.plot(recall, precision, label=f'{name} (PR AUC = {pr_auc:.3f})', linewidth=2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Courbes Precision-Recall (plus pertinent que ROC pour classes déséquilibrées)')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.savefig('precision_recall_curves.png', dpi=100, bbox_inches='tight')
plt.show()
print(" PR AUC curves sauvegardées: 'precision_recall_curves.png'")

# ------------------------------
# 9. COURBES DE CALIBRATION (Bonus)
# ------------------------------
print("\n" + "="*60)
print("8. COURBES DE CALIBRATION")
print("="*60)

plt.figure(figsize=(10, 8))

for name, model in models.items():
    if name == 'SVM':
        continue  # SVM avec probability=True parfois instable pour calibration
    best_model = models[name]
    rs = RandomizedSearchCV(best_model, param_distributions.get(name, {}), n_iter=10, cv=cv_strategy, random_state=42)
    rs.fit(X_train_scaled, y_train)
    y_proba = rs.best_estimator_.predict_proba(X_test_scaled)[:, 1]
    
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=name, linewidth=2)

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Parfaitement calibré')
plt.xlabel('Probabilité prédite moyenne')
plt.ylabel('Fraction de positifs')
plt.title('Courbes de Calibration (fiabilité des probabilités)')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.savefig('calibration_curves.png', dpi=100, bbox_inches='tight')
plt.show()
print(" Calibration curves sauvegardées: 'calibration_curves.png'")

# ------------------------------
# 10. LEARNING CURVES (Détection Overfitting/Underfitting)
# ------------------------------
print("\n" + "="*60)
print("9. LEARNING CURVES")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, model) in enumerate(models.items()):
    if idx >= 6:
        break
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train_scaled, y_train, cv=cv_strategy,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='f1', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    axes[idx].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    axes[idx].plot(train_sizes, train_mean, 'o-', color='blue', label='Train')
    axes[idx].plot(train_sizes, test_mean, 'o-', color='orange', label='Validation')
    axes[idx].set_title(f'Learning Curve - {name}')
    axes[idx].set_xlabel('Training examples')
    axes[idx].set_ylabel('F1 Score')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('learning_curves.png', dpi=100, bbox_inches='tight')
plt.show()
print(" Learning curves sauvegardées: 'learning_curves.png'")

# ------------------------------
# SHAP VALUES - Version corrigée pour array 3D
# ------------------------------
print("\n" + "="*60)
print("10. SHAP VALUES - INTERPRÉTABILITÉ (XAI)")
print("="*60)

# On prend le meilleur modèle (Random Forest)
best_rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=30, min_samples_split=5, min_samples_leaf=2)
best_rf.fit(X_train_scaled, y_train)

# Création de l'explainer SHAP
explainer = shap.TreeExplainer(best_rf)

# Calcul des valeurs SHAP
shap_values = explainer.shap_values(X_test_scaled)

# Vérification des dimensions
print(f"Shape de X_test_scaled: {X_test_scaled.shape}")
print(f"Shape de shap_values: {shap_values.shape}")

# CORRECTION ICI: shap_values[:, :, 1] au lieu de shap_values[1]
# Pour array 3D (n_samples, n_features, n_classes)
shap_values_class1 = shap_values[:, :, 1]  # Classe 1 (bénin)

print(f"Shape des SHAP values pour classe 1: {shap_values_class1.shape}")
print(f"Vérification: {shap_values_class1.shape} == {X_test_scaled.shape}")

# Summary plot (beeswarm)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_class1, X_test_scaled, 
                 feature_names=data.feature_names, 
                 show=False, max_display=15)
plt.tight_layout()
plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print(" SHAP summary plot sauvegardé: 'shap_summary_plot.png'")

# Bar plot (importance globale)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_class1, X_test_scaled, 
                 feature_names=data.feature_names, 
                 plot_type="bar", show=False, max_display=15)
plt.tight_layout()
plt.savefig('shap_bar_plot.png', dpi=150, bbox_inches='tight')
plt.close()
print(" SHAP bar plot sauvegardé: 'shap_bar_plot.png'")

# Calcul de l'importance moyenne absolue
mean_abs_shap = np.abs(shap_values_class1).mean(axis=0)
feature_importance = pd.DataFrame({
    'caractéristique': data.feature_names,
    'importance_shap': mean_abs_shap
}).sort_values('importance_shap', ascending=False)

print("\nTop 10 des caractéristiques (SHAP):")
for i in range(10):
    print(f"   {i+1}. {feature_importance.iloc[i]['caractéristique']}: {feature_importance.iloc[i]['importance_shap']:.4f}")

# Sauvegarder
feature_importance.to_csv('shap_feature_importance.csv', index=False)
# ------------------------------
# 12. ENSEMBLE LEARNING AVANCÉ (Stacking - Bonus)
# ------------------------------
print("\n" + "="*60)
print("11. ENSEMBLE LEARNING AVANCÉ - STACKING")
print("="*60)

# Création d'un stacking classifier
base_learners = [
    ('lr', LogisticRegression(random_state=42, max_iter=1000)),
    ('knn', KNeighborsClassifier(n_neighbors=7)),
    ('svm', SVC(probability=True, random_state=42))
]

meta_learner = RandomForestClassifier(random_state=42, n_estimators=100)

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    stack_method='predict_proba'
)

# Entraînement
stacking_clf.fit(X_train_scaled, y_train)
y_pred_stack = stacking_clf.predict(X_test_scaled)
y_proba_stack = stacking_clf.predict_proba(X_test_scaled)[:, 1]

# Évaluation
print("\n Résultats du Stacking Classifier:")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_stack):.4f}")
print(f"   F1-Score: {f1_score(y_test, y_pred_stack):.4f}")
print(f"   ROC AUC: {roc_auc_score(y_test, y_proba_stack):.4f}")
print(f"   MCC: {matthews_corrcoef(y_test, y_pred_stack):.4f}")

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(random_state=42, n_estimators=100)),
        ('xgb', XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')),
        ('lr', LogisticRegression(random_state=42, max_iter=1000))
    ],
    voting='soft'
)

voting_clf.fit(X_train_scaled, y_train)
y_pred_vote = voting_clf.predict(X_test_scaled)

print("\nRésultats du Voting Classifier (Soft):")
print(f"   Accuracy: {accuracy_score(y_test, y_pred_vote):.4f}")
print(f"   F1-Score: {f1_score(y_test, y_pred_vote):.4f}")

# ------------------------------
# 13. ANALYSE DE L'IMPACT DU SCALING
# ------------------------------
print("\n" + "="*60)
print("12. ANALYSE DE L'IMPACT DU SCALING")
print("="*60)

comparison_scaling = []

for name, model in models.items():
    # Sans scaling
    model_no_scaler = model
    model_no_scaler.fit(X_train, y_train)
    y_pred_no = model_no_scaler.predict(X_test)
    f1_no = f1_score(y_test, y_pred_no)
    
    # Avec scaling (déjà calculé)
    rs = RandomizedSearchCV(model, param_distributions.get(name, {}), n_iter=10, cv=cv_strategy, random_state=42)
    rs.fit(X_train_scaled, y_train)
    y_pred_scaled = rs.best_estimator_.predict(X_test_scaled)
    f1_scaled = f1_score(y_test, y_pred_scaled)
    
    comparison_scaling.append({
        'Modèle': name,
        'F1 sans scaling': f1_no,
        'F1 avec scaling': f1_scaled,
        'Différence': f1_scaled - f1_no
    })

scaling_df = pd.DataFrame(comparison_scaling)
print("\n Impact du StandardScaler sur les performances F1:")
print(scaling_df.to_string(index=False))

# ------------------------------
# 14. RAPPORT FINAL
# ------------------------------
print("\n" + "="*60)
print("13. RAPPORT FINAL - SYNTHÈSE")
print("="*60)

# Meilleur modèle
best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Modèle']
best_f1 = results_df['F1-Score'].max()

# Modèle le plus stable (plus petit écart-type CV)
most_stable = results_df.loc[results_df['CV F1 Std'].idxmin(), 'Modèle']

print(f"\n MEILLEUR MODÈLE (F1-Score): {best_model_name} (F1 = {best_f1:.4f})")
print(f" MODÈLE LE PLUS STABLE: {most_stable} (Std = {results_df['CV F1 Std'].min():.4f})")

