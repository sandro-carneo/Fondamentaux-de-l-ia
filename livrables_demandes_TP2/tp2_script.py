import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import shap
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("ETAPE 2 - Préparation des données (Cas A - Churn)")
print("="*50)

# Chargement depuis URL publique
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df  = pd.read_csv(url)
print(f"Forme : {df.shape}")
print(f"\nDistribution cible :")
print(df['Churn'].value_counts())
print(f"\nColonnes : {list(df.columns)}")

# Nettoyage : TotalCharges contient des espaces
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
print(f"\nAprès nettoyage : {df.shape}")

# Encodage : variables catégorielles en dummies
df_enc = pd.get_dummies(df.drop('customerID', axis=1), drop_first=True)
X = df_enc.drop('Churn_Yes', axis=1)
y = df_enc['Churn_Yes'].astype(int)
feature_names = list(X.columns)

# Déséquilibre de classes
print(f"\nTaux de churn : {y.mean()*100:.1f}% -> dataset déséquilibré")
print("-> Utiliser F1-macro plutôt qu'accuracy seule")

# Split stratifié
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalisation
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"\nTrain : {len(X_train)} | Test : {len(X_test)}")
print(f"Nombre de features : {len(feature_names)}")


print("\n" + "="*50)
print("ETAPE 3 - Modélisation : 3 modèles à comparer")
print("="*50)

modeles = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(
                               n_estimators=100, random_state=42,
                               eval_metric='logloss', verbosity=0
                           ),
}

resultats = {}
for nom, modele in modeles.items():
    modele.fit(X_train_sc, y_train)
    pred   = modele.predict(X_test_sc)
    acc    = accuracy_score(y_test, pred)
    f1_mac = f1_score(y_test, pred, average='macro')
    f1_wei = f1_score(y_test, pred, average='weighted')
    resultats[nom] = {'accuracy': acc, 'f1_macro': f1_mac, 'f1_weighted': f1_wei}
    print(f"{nom:22s} -> Accuracy : {acc*100:.1f}%  |  F1-macro : {f1_mac:.3f}  |  F1-weighted : {f1_wei:.3f}")

# Tableau comparatif
df_resultats = pd.DataFrame(resultats).T
print("\n=== TABLEAU COMPARATIF ===")
print(df_resultats.round(3))

# Visualisation
df_resultats[['accuracy', 'f1_macro']].plot(kind='bar', figsize=(9, 4))
plt.title('Comparaison des modèles')
plt.ylabel('Score')
plt.xticks(rotation=20)
plt.ylim(0.5, 1.0)
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.savefig('comparaison_modeles.png')
plt.close()
print("\n[+] Graphique de comparaison des modèles sauvegardé : comparaison_modeles.png")


print("\n" + "="*50)
print("ETAPE 4 - Évaluation approfondie")
print("="*50)

NOM_MEILLEUR = "Logistic Regression"
meilleur = modeles[NOM_MEILLEUR]
y_pred   = meilleur.predict(X_test_sc)

# Rapport complet
print(f"=== RAPPORT DÉTAILLÉ - {NOM_MEILLEUR} ===")
print(classification_report(y_test, y_pred))

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f'Matrice de confusion - {NOM_MEILLEUR}')
plt.ylabel('Réalité')
plt.xlabel('Prédiction')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
print("[+] Matrice de confusion sauvegardée : confusion_matrix.png")

# Évolution de Random Forest
n_estimators_range = [10, 25, 50, 100, 200]
scores_rf = []
for n in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n, random_state=42)
    rf_temp.fit(X_train_sc, y_train)
    pred_temp = rf_temp.predict(X_test_sc)
    scores_rf.append(f1_score(y_test, pred_temp, average='macro'))

plt.figure(figsize=(8, 4))
plt.plot(n_estimators_range, scores_rf, 'g-o')
plt.xlabel("Nombre d'arbres (n_estimators)")
plt.ylabel("F1-score macro")
plt.title("Évolution des performances - Random Forest")
plt.grid(True)
plt.tight_layout()
plt.savefig('rf_evolution.png')
plt.close()
print("[+] Graphique d'évolution RF sauvegardé : rf_evolution.png")


print("\n" + "="*50)
print("ETAPE 5 - Explicabilité SHAP")
print("="*50)

rf_model = modeles["Random Forest"]
X_sample = X_test_sc[:200]
if sp.issparse(X_sample):
    X_sample = X_sample.toarray()
else:
    X_sample = np.array(X_sample)

explainer   = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    sv = shap_values[1]
elif len(shap_values.shape) == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

plt.figure()
shap.summary_plot(
    sv, X_sample,
    feature_names=feature_names,
    max_display=15,
    show=False
)
plt.title("SHAP - Top 15 variables les plus influentes (classe positive)")
plt.tight_layout()
plt.savefig('shap_summary.png', bbox_inches='tight')
plt.close()
print("[+] Graphique SHAP sauvegardé : shap_summary.png")
print("\n>>> Exécution terminée avec succès ! Toutes les informations et images sont générées.")
