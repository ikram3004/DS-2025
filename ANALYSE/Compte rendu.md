# COMPTE RENDU : HOUSE PRICES DATASET

## TABLE DES MATIÈRES

1. [Introduction](#1-introduction)
2. [Description du Dataset](#2-description-du-dataset)
3. [Localisation Géographique](#3-localisation-géographique)
4. [Analyse Exploratoire des Données](#4-analyse-exploratoire-des-données)
5. [Visualisations Graphiques](#5-visualisations-graphiques)
6. [Matrice de Corrélation](#6-matrice-de-corrélation)
7. [Modèles de Régression](#7-modèles-de-régression)
8. [Conclusion](#8-conclusion)

---

## 1. INTRODUCTION

Le **House Prices Dataset** est un jeu de données provenant d'une compétition Kaggle intitulée "House Prices: Advanced Regression Techniques". L'objectif principal est de prédire le prix de vente final des maisons résidentielles basé sur 79 variables explicatives.

**Problématique** : Comment prédire avec précision le prix d'une maison en fonction de ses caractéristiques physiques, sa localisation et ses équipements ?

URL : https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques 
---

## 2. DESCRIPTION DU DATASET

### Caractéristiques générales
- **Nombre d'observations (train)** : 1460 maisons
- **Nombre d'observations (test)** : 1459 maisons
- **Nombre de variables** : 81 (79 features + Id + SalePrice)
- **Variable cible** : SalePrice (prix de vente en dollars)

### Types de variables
- **Variables numériques** : 38 (superficie, année, nombre de pièces, etc.)
- **Variables catégorielles** : 43 (qualité, type de maison, quartier, etc.)

### Principales variables
- **OverallQual** : Qualité générale de la maison (1-10)
- **GrLivArea** : Surface habitable au-dessus du sol (pieds carrés)
- **GarageCars** : Capacité du garage en nombre de voitures
- **TotalBsmtSF** : Surface totale du sous-sol (pieds carrés)
- **YearBuilt** : Année de construction
- **Neighborhood** : Quartier où se situe la maison

---

## 3. LOCALISATION GÉOGRAPHIQUE

### Lieu réel du dataset
**Ville** : Ames, Iowa, États-Unis

**Coordonnées géographiques** :
- Latitude : 42.0308° N
- Longitude : 93.6319° O

**Contexte géographique** :
- Ames est une ville universitaire de l'Iowa, située dans le Midwest américain
- Population : environ 66 000 habitants
- Abrite l'Iowa State University
- Économie basée sur l'éducation, la recherche et l'agriculture
- Marché immobilier représentatif des villes moyennes américaines

**Période de collecte** : Les données couvrent les ventes de maisons de 2006 à 2010.

**Caractéristiques du marché local** :
- Prix médian des maisons : environ $180,000
- Marché relativement stable comparé aux grandes métropoles
- Diversité architecturale : maisons de style victorien, ranch, colonial

---

## 4. ANALYSE EXPLORATOIRE DES DONNÉES

### Statistiques descriptives de SalePrice

```
Moyenne        : $180,921
Médiane        : $163,000
Écart-type     : $79,442
Minimum        : $34,900
Maximum        : $755,000
```

### Distribution du prix de vente
- Distribution asymétrique vers la droite (skewness positif)
- Présence de valeurs extrêmes (maisons de luxe)
- La plupart des maisons se vendent entre $100,000 et $250,000

### Valeurs manquantes principales
- **PoolQC** : 99.5% manquant (peu de maisons ont une piscine)
- **MiscFeature** : 96.3% manquant
- **Alley** : 93.8% manquant
- **Fence** : 80.8% manquant
- **FireplaceQu** : 47.3% manquant

---

## 5. VISUALISATIONS GRAPHIQUES

### Code Python pour les graphiques

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Chargement des données
train = pd.read_csv('train.csv')

# Configuration style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)

# 1. DISTRIBUTION DU PRIX DE VENTE
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
sns.histplot(train['SalePrice'], kde=True, bins=50, color='skyblue')
plt.title('Distribution du Prix de Vente', fontsize=14, fontweight='bold')
plt.xlabel('Prix de Vente ($)')
plt.ylabel('Fréquence')

plt.subplot(1, 2, 2)
stats.probplot(train['SalePrice'], dist="norm", plot=plt)
plt.title('Q-Q Plot du Prix de Vente', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 2. TOP 10 DES VARIABLES CORRELÉES AVEC SALEPRICE
correlations = train.corr()['SalePrice'].sort_values(ascending=False)
top_features = correlations[1:11]

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
plt.title('Top 10 des Variables Corrélées avec SalePrice', fontsize=14, fontweight='bold')
plt.xlabel('Coefficient de Corrélation')
plt.tight_layout()
plt.show()

# 3. RELATION ENTRE SURFACE HABITABLE ET PRIX
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train, x='GrLivArea', y='SalePrice', alpha=0.6, color='coral')
plt.title('Prix de Vente vs Surface Habitable', fontsize=14, fontweight='bold')
plt.xlabel('Surface Habitable (pieds carrés)')
plt.ylabel('Prix de Vente ($)')
plt.tight_layout()
plt.show()

# 4. PRIX PAR QUALITÉ GÉNÉRALE
plt.figure(figsize=(12, 6))
sns.boxplot(data=train, x='OverallQual', y='SalePrice', palette='Set2')
plt.title('Prix de Vente par Qualité Générale', fontsize=14, fontweight='bold')
plt.xlabel('Qualité Générale (1-10)')
plt.ylabel('Prix de Vente ($)')
plt.tight_layout()
plt.show()

# 5. PRIX PAR QUARTIER (TOP 10)
neighborhood_prices = train.groupby('Neighborhood')['SalePrice'].median().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x=neighborhood_prices.values, y=neighborhood_prices.index, palette='coolwarm')
plt.title('Top 10 Quartiers par Prix Médian', fontsize=14, fontweight='bold')
plt.xlabel('Prix Médian ($)')
plt.ylabel('Quartier')
plt.tight_layout()
plt.show()

# 6. ÉVOLUTION DES PRIX PAR ANNÉE DE CONSTRUCTION
plt.figure(figsize=(12, 6))
year_prices = train.groupby('YearBuilt')['SalePrice'].mean()
plt.plot(year_prices.index, year_prices.values, linewidth=2, color='darkblue')
plt.title('Prix Moyen par Année de Construction', fontsize=14, fontweight='bold')
plt.xlabel('Année de Construction')
plt.ylabel('Prix Moyen ($)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 6. MATRICE DE CORRÉLATION

### Code Python pour la matrice de corrélation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Chargement des données
train = pd.read_csv('train.csv')

# Sélection des variables numériques
numerical_features = train.select_dtypes(include=[np.number]).columns.tolist()

# 1. MATRICE DE CORRÉLATION COMPLÈTE
plt.figure(figsize=(16, 14))
correlation_matrix = train[numerical_features].corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot=False)
plt.title('Matrice de Corrélation - Toutes Variables Numériques', 
          fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

# 2. MATRICE DE CORRÉLATION - TOP 15 VARIABLES
top_corr_features = correlation_matrix['SalePrice'].abs().sort_values(ascending=False).head(16).index

plt.figure(figsize=(12, 10))
sns.heatmap(train[top_corr_features].corr(), annot=True, fmt='.2f', 
            cmap='RdYlGn', center=0, square=True, linewidths=1,
            cbar_kws={"shrink": 0.8})
plt.title('Matrice de Corrélation - Top 15 Variables', 
          fontsize=14, fontweight='bold', pad=15)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# 3. ANALYSE DES CORRÉLATIONS AVEC SALEPRICE
print("=" * 60)
print("CORRÉLATIONS AVEC SALEPRICE")
print("=" * 60)
correlations_with_price = correlation_matrix['SalePrice'].sort_values(ascending=False)

print("\n🔼 TOP 10 CORRÉLATIONS POSITIVES :")
print(correlations_with_price.head(11))

print("\n🔽 TOP 10 CORRÉLATIONS NÉGATIVES :")
print(correlations_with_price.tail(10))

# 4. DÉTECTION DE MULTICOLINÉARITÉ
print("\n" + "=" * 60)
print("DÉTECTION DE MULTICOLINÉARITÉ")
print("=" * 60)

# Paires de variables avec corrélation > 0.8 (hors diagonale)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append({
                'Variable 1': correlation_matrix.columns[i],
                'Variable 2': correlation_matrix.columns[j],
                'Corrélation': correlation_matrix.iloc[i, j]
            })

if high_corr_pairs:
    multicolinearity_df = pd.DataFrame(high_corr_pairs)
    print("\n⚠️ Paires de variables fortement corrélées (|r| > 0.8) :")
    print(multicolinearity_df.to_string(index=False))
else:
    print("\n✓ Aucune multicolinéarité forte détectée")
```

### Principales corrélations observées

**Corrélations positives fortes avec SalePrice** :
1. OverallQual (0.79) - Qualité générale
2. GrLivArea (0.71) - Surface habitable
3. GarageCars (0.64) - Capacité garage
4. GarageArea (0.62) - Surface garage
5. TotalBsmtSF (0.61) - Surface sous-sol

**Corrélations négatives** :
- Les corrélations négatives sont généralement faibles
- Aucune variable ne montre une corrélation négative forte

---

## 7. MODÈLES DE RÉGRESSION

### Code Python pour les modèles

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# ========================================
# 1. PRÉPARATION DES DONNÉES
# ========================================

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Séparation features et target
X = train.drop(['SalePrice', 'Id'], axis=1)
y = train['SalePrice']
test_ids = test['Id']
X_test = test.drop('Id', axis=1)

# Gestion des valeurs manquantes
# Pour les variables numériques : remplir avec la médiane
numeric_features = X.select_dtypes(include=[np.number]).columns
for col in numeric_features:
    X[col].fillna(X[col].median(), inplace=True)
    X_test[col].fillna(X_test[col].median(), inplace=True)

# Pour les variables catégorielles : remplir avec 'None'
categorical_features = X.select_dtypes(include=['object']).columns
for col in categorical_features:
    X[col].fillna('None', inplace=True)
    X_test[col].fillna('None', inplace=True)

# Encodage des variables catégorielles
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print("✓ Préparation des données terminée")
print(f"  - Forme X_train: {X_train.shape}")
print(f"  - Forme X_val: {X_val.shape}")
print(f"  - Forme y_train: {y_train.shape}")

# ========================================
# 2. ENTRAÎNEMENT DES MODÈLES
# ========================================

models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=10.0),
    'Lasso': Lasso(alpha=100.0),
    'ElasticNet': ElasticNet(alpha=100.0, l1_ratio=0.5),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
}

results = []

print("\n" + "="*80)
print("ENTRAÎNEMENT ET ÉVALUATION DES MODÈLES")
print("="*80)

for name, model in models.items():
    print(f"\n📊 Entraînement : {name}...")
    
    # Entraînement
    if name in ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
    
    # Métriques
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    results.append({
        'Modèle': name,
        'RMSE': rmse,
        'MAE': mae,
        'R² Score': r2
    })
    
    print(f"  ✓ RMSE: ${rmse:,.2f}")
    print(f"  ✓ MAE: ${mae:,.2f}")
    print(f"  ✓ R² Score: {r2:.4f}")

# ========================================
# 3. COMPARAISON DES RÉSULTATS
# ========================================

results_df = pd.DataFrame(results).sort_values('RMSE')

print("\n" + "="*80)
print("TABLEAU RÉCAPITULATIF DES PERFORMANCES")
print("="*80)
print(results_df.to_string(index=False))

# Visualisation des performances
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# RMSE
axes[0].barh(results_df['Modèle'], results_df['RMSE'], color='coral')
axes[0].set_xlabel('RMSE ($)')
axes[0].set_title('Root Mean Squared Error', fontweight='bold')
axes[0].invert_yaxis()

# MAE
axes[1].barh(results_df['Modèle'], results_df['MAE'], color='skyblue')
axes[1].set_xlabel('MAE ($)')
axes[1].set_title('Mean Absolute Error', fontweight='bold')
axes[1].invert_yaxis()

# R² Score
axes[2].barh(results_df['Modèle'], results_df['R² Score'], color='lightgreen')
axes[2].set_xlabel('R² Score')
axes[2].set_title('Coefficient de Détermination', fontweight='bold')
axes[2].invert_yaxis()
axes[2].set_xlim([0, 1])

plt.tight_layout()
plt.show()

# ========================================
# 4. PRÉDICTIONS AVEC LE MEILLEUR MODÈLE
# ========================================

best_model_name = results_df.iloc[0]['Modèle']
best_model = models[best_model_name]

print(f"\n🏆 Meilleur modèle : {best_model_name}")

# Graphique : Valeurs réelles vs prédites
if best_model_name in ['Linear Regression', 'Ridge', 'Lasso', 'ElasticNet']:
    y_pred_best = best_model.predict(X_val_scaled)
else:
    y_pred_best = best_model.predict(X_val)

plt.figure(figsize=(10, 6))
plt.scatter(y_val, y_pred_best, alpha=0.6, color='purple')
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Prix Réels ($)')
plt.ylabel('Prix Prédits ($)')
plt.title(f'Prédictions vs Réalité - {best_model_name}', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.show()

# Distribution des résidus
residuals = y_val - y_pred_best

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(residuals, kde=True, bins=50, color='teal')
plt.xlabel('Résidus ($)')
plt.ylabel('Fréquence')
plt.title('Distribution des Résidus', fontweight='bold')

plt.subplot(1, 2, 2)
plt.scatter(y_pred_best, residuals, alpha=0.6, color='orange')
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Prix Prédits ($)')
plt.ylabel('Résidus ($)')
plt.title('Résidus vs Prédictions', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n✓ Analyse de régression terminée")
```

### Résultats attendus

Les modèles basés sur les ensembles (Random Forest, Gradient Boosting, XGBoost) obtiennent généralement les meilleures performances avec :
- **RMSE** : entre $25,000 et $35,000
- **R² Score** : entre 0.85 et 0.92
- **MAE** : entre $15,000 et $20,000

---

## 8. CONCLUSION

### Points clés
✅ **Dataset riche** : 79 variables permettent une modélisation détaillée
✅ **Corrélations identifiées** : La qualité générale et la surface habitable sont les prédicteurs les plus forts
✅ **Modèles performants** : Les algorithmes d'ensemble (XGBoost, Random Forest) surpassent les modèles linéaires
✅ **Localisation réelle** : Données provenant d'Ames, Iowa (2006-2010)

### Recommandations
1. **Feature Engineering** : Créer des variables dérivées (surface totale, âge de la maison)
2. **Traitement des outliers** : Supprimer les valeurs extrêmes pour améliorer les prédictions
3. **Optimisation des hyperparamètres** : Utiliser GridSearchCV ou RandomizedSearchCV
4. **Transformation de la variable cible** : Appliquer log(SalePrice) pour normaliser la distribution
5. **Validation croisée** : Utiliser K-Fold pour une évaluation plus robuste

### Applications pratiques
- Estimation automatique de prix pour agences immobilières
- Aide à la décision pour acheteurs et vendeurs
- Analyse de marché immobilier
- Détection de bonnes affaires (maisons sous-évaluées)

---

**Auteur** : Analyse réalisée dans le cadre de l'étude du dataset House Prices  
**Date** : Novembre 2025  
**Source** : Kaggle - House Prices: Advanced Regression Techniques
