# Rapport Académique : Analyse Exploratoire du Dataset Wine Quality
EZ-ZEMANY IKRAM

---

## Résumé

Ce rapport présente une analyse exploratoire du jeu de données **Wine Quality** issu du UCI Machine Learning Repository. L'objectif principal est d'examiner les caractéristiques physicochimiques des vins et d'identifier les corrélations entre ces variables afin de mieux comprendre les facteurs influençant la qualité du vin.

---

## 1. Introduction

### 1.1 Contexte

L'industrie viticole s'appuie de plus en plus sur des méthodes analytiques pour évaluer et prédire la qualité des vins. Le dataset Wine Quality, créé par P. Cortez et al. (2009), constitue une référence dans le domaine du machine learning pour les tâches de classification et de régression appliquées à l'œnologie.

### 1.2 Objectifs

- Charger et explorer le dataset Wine Quality depuis le UCI Repository
- Analyser les variables physicochimiques disponibles
- Calculer et visualiser la matrice de corrélation entre les variables
- Identifier les relations significatives pouvant influencer la qualité du vin

---

## 2. Matériel et Méthodes

### 2.1 Source des Données

Les données proviennent du **UCI Machine Learning Repository** (id=186) et sont accessibles via le package Python `ucimlrepo`. Ce dataset combine des échantillons de vins rouges et blancs portugais de la région "Vinho Verde".

### 2.2 Outils et Bibliothèques

L'analyse a été réalisée en Python avec les bibliothèques suivantes :

```python
pip install ucimlrepo
```

```python
from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

### 2.3 Chargement des Données

Le code suivant permet de récupérer le dataset depuis le repository UCI :

```python
# Charger le dataset Wine Quality (id=186)
wine_quality = fetch_ucirepo(id=186)

# Dataframes
X = wine_quality.data.features
y = wine_quality.data.targets

# Affichage infos du dataset
print("=== METADATA ===")
print(wine_quality.metadata)

print("\n=== VARIABLES ===")
print(wine_quality.variables)
```

---

## 3. Description de la Base de Données

### 3.1 Caractéristiques Générales

| Attribut | Valeur |
|----------|--------|
| **Nom** | Wine Quality |
| **Nombre d'instances** | 6 497 (1 599 rouges + 4 898 blancs) |
| **Nombre de variables** | 12 (11 features + 1 target) |
| **Type de tâche** | Classification / Régression |
| **Valeurs manquantes** | Non |

### 3.2 Variables Explicatives (Features)

Le dataset contient **11 variables physicochimiques** :

| Variable | Description | Unité |
|----------|-------------|-------|
| **fixed acidity** | Acidité fixe (acides tartriques) | g/dm³ |
| **volatile acidity** | Acidité volatile (acide acétique) | g/dm³ |
| **citric acid** | Acide citrique | g/dm³ |
| **residual sugar** | Sucre résiduel | g/dm³ |
| **chlorides** | Chlorures (sel) | g/dm³ |
| **free sulfur dioxide** | Dioxyde de soufre libre | mg/dm³ |
| **total sulfur dioxide** | Dioxyde de soufre total | mg/dm³ |
| **density** | Densité | g/cm³ |
| **pH** | Niveau d'acidité | - |
| **sulphates** | Sulfates | g/dm³ |
| **alcohol** | Teneur en alcool | % vol. |

### 3.3 Variable Cible (Target)

| Variable | Description | Plage |
|----------|-------------|-------|
| **quality** | Score de qualité attribué par des experts | 0 à 10 |

La variable `quality` représente une évaluation sensorielle réalisée par au moins trois experts œnologues. Le score médian des évaluations est retenu comme valeur finale.

---

## 4. Analyse des Corrélations

### 4.1 Méthodologie

La matrice de corrélation de Pearson a été calculée pour identifier les relations linéaires entre toutes les variables du dataset :

```python
# Fusionner X et y dans un seul DataFrame
df = pd.concat([X, y], axis=1)

# Calculer la matrice de corrélation
corr_matrix = df.corr()

# Afficher la matrice dans la console
print("\n=== MATRICE DE CORRÉLATION ===")
print(corr_matrix)
```

### 4.2 Visualisation

Une heatmap a été générée pour représenter visuellement les corrélations :

```python
# Visualisation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Matrice de corrélation - Wine Quality Dataset")
plt.show()
```

### 4.3 Interprétation des Résultats

#### Corrélations Positives Notables

- **alcohol ↔ quality** : Corrélation positive modérée, suggérant que les vins avec une teneur en alcool plus élevée tendent à recevoir de meilleurs scores
- **citric acid ↔ fixed acidity** : Forte corrélation positive entre ces deux types d'acidité
- **free sulfur dioxide ↔ total sulfur dioxide** : Corrélation logique puisque le SO₂ libre fait partie du SO₂ total

#### Corrélations Négatives Notables

- **volatile acidity ↔ quality** : Corrélation négative, indiquant qu'une acidité volatile élevée (vinaigre) dégrade la qualité perçue
- **density ↔ alcohol** : Corrélation négative forte, car l'alcool est moins dense que l'eau
- **pH ↔ fixed acidity** : Relation inverse attendue chimiquement

---

## 5. Discussion

### 5.1 Observations Principales

L'analyse révèle que la qualité du vin est principalement influencée par la teneur en alcool (positivement) et l'acidité volatile (négativement). Ces résultats sont cohérents avec la littérature œnologique : un vin avec trop d'acide acétique présente des défauts organoleptiques.

### 5.2 Limites de l'Étude

- L'analyse se limite aux corrélations linéaires (Pearson)
- Les interactions non-linéaires entre variables ne sont pas explorées
- La distinction entre vins rouges et blancs n'a pas été prise en compte dans cette analyse

### 5.3 Perspectives

Pour approfondir cette étude, il serait pertinent de :
- Appliquer des algorithmes de classification (Random Forest, SVM, etc.)
- Réaliser une analyse en composantes principales (ACP)
- Séparer l'analyse par type de vin (rouge vs blanc)

---

## 6. Conclusion

Cette analyse exploratoire du dataset Wine Quality a permis d'identifier les principales corrélations entre les caractéristiques physicochimiques et la qualité du vin. Les variables **alcohol** et **volatile acidity** apparaissent comme les facteurs les plus déterminants. Ces résultats constituent une base solide pour le développement de modèles prédictifs de la qualité du vin.

---

## Références

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). Modeling wine preferences by data mining from physicochemical properties. *Decision Support Systems*, 47(4), 547-553.
- UCI Machine Learning Repository : [https://archive.ics.uci.edu/dataset/186/wine+quality](https://archive.ics.uci.edu/dataset/186/wine+quality)

---

## Annexe : Code Source Complet

```python
from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Charger le dataset Wine Quality (id=186)
wine_quality = fetch_ucirepo(id=186)

# Dataframes
X = wine_quality.data.features
y = wine_quality.data.targets

# Affichage infos du dataset
print("=== METADATA ===")
print(wine_quality.metadata)

print("\n=== VARIABLES ===")
print(wine_quality.variables)

# Fusionner X et y dans un seul DataFrame
df = pd.concat([X, y], axis=1)

# Calculer la matrice de corrélation
corr_matrix = df.corr()

# Afficher la matrice dans la console
print("\n=== MATRICE DE CORRÉLATION ===")
print(corr_matrix)

# Visualisation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Matrice de corrélation - Wine Quality Dataset")
plt.show()
```
<img width="1029" height="804" alt="image" src="https://github.com/user-attachments/assets/19d3f46b-979c-4fdb-97db-494b0b651671" />
