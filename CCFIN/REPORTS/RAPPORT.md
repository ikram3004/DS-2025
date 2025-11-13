
# Rapport d'Étude : Student Performance Dataset

Rapport réalisé par: EZ-ZEMANY IKRAM

## 1. Introduction

### 1.1 Contexte de l'étude
Le Student Performance Dataset est une base de données issue d'une recherche menée à l'Université de Minho au Portugal, visant à analyser et prédire la performance académique des étudiants de l'enseignement secondaire. Cette étude s'inscrit dans le domaine de l'Educational Data Mining (EDM), qui utilise les techniques d'apprentissage automatique pour améliorer les systèmes éducatifs.

### 1.2 Provenance des données
- **Source** : UCI Machine Learning Repository (Dataset ID: 320)
- **Auteurs** : Paulo Cortez et Alice Silva
- **Institution** : Université de Minho, Portugal
- **Année de publication** : 2008
- **Établissements concernés** : Deux écoles secondaires portugaises (Gabriel Pereira et Mousinho da Silveira)

### 1.3 Objectifs de la recherche
L'objectif principal de cette étude est d'identifier les facteurs déterminants de la réussite scolaire et de développer des modèles prédictifs capables d'anticiper les performances académiques des étudiants. Cette approche permet de mettre en place des systèmes d'alerte précoce et d'interventions ciblées pour améliorer les résultats scolaires.

---

## 2. Description du Dataset

### 2.1 Composition générale
Le dataset comprend deux ensembles de données distincts correspondant à deux disciplines :

| Discipline | Nombre d'étudiants | École |
|------------|-------------------|-------|
| Mathématiques | 395 | Gabriel Pereira (GP) et Mousinho da Silveira (MS) |
| Portugais | 649 | Gabriel Pereira (GP) et Mousinho da Silveira (MS) |
| **Total** | **~1044** | **2 établissements** |

### 2.2 Structure des données
Le dataset contient **33 attributs** répartis en plusieurs catégories, ainsi que **3 variables cibles** représentant les notes obtenues à différentes périodes de l'année scolaire.

---

## 3. Variables et Attributs

### 3.1 Variables démographiques

| Variable | Description | Valeurs possibles |
|----------|-------------|-------------------|
| **age** | Âge de l'étudiant | 15 à 22 ans |
| **sex** | Sexe de l'étudiant | F (Féminin), M (Masculin) |
| **address** | Type de résidence | U (Urbain), R (Rural) |
| **famsize** | Taille de la famille | LE3 (≤3 personnes), GT3 (>3 personnes) |
| **Pstatus** | Statut cohabitant des parents | T (Together - ensemble), A (Apart - séparés) |

### 3.2 Variables socio-économiques

| Variable | Description | Échelle |
|----------|-------------|---------|
| **Medu** | Niveau d'éducation de la mère | 0 (aucun) à 4 (enseignement supérieur) |
| **Fedu** | Niveau d'éducation du père | 0 (aucun) à 4 (enseignement supérieur) |
| **Mjob** | Profession de la mère | teacher, health, services, at_home, other |
| **Fjob** | Profession du père | teacher, health, services, at_home, other |
| **guardian** | Tuteur légal principal | mother, father, other |
| **traveltime** | Temps de trajet domicile-école | 1 (<15 min) à 4 (>1 heure) |

### 3.3 Variables liées au support éducatif

| Variable | Description | Type |
|----------|-------------|------|
| **schoolsup** | Support éducatif supplémentaire de l'école | Binaire (yes/no) |
| **famsup** | Support éducatif familial | Binaire (yes/no) |
| **paid** | Cours particuliers payants | Binaire (yes/no) |
| **activities** | Participation aux activités extrascolaires | Binaire (yes/no) |
| **nursery** | Fréquentation de la maternelle | Binaire (yes/no) |
| **higher** | Désir de poursuivre des études supérieures | Binaire (yes/no) |
| **internet** | Accès internet à domicile | Binaire (yes/no) |

### 3.4 Variables de comportement académique

| Variable | Description | Échelle/Valeurs |
|----------|-------------|-----------------|
| **studytime** | Temps d'étude hebdomadaire | 1 (<2h) à 4 (>10h) |
| **failures** | Nombre d'échecs scolaires antérieurs | 0 à 4 (4 = 4 ou plus) |
| **absences** | Nombre d'absences durant l'année | 0 à 93 |

### 3.5 Variables comportementales et sociales

| Variable | Description | Échelle |
|----------|-------------|---------|
| **freetime** | Temps libre après l'école | 1 (très faible) à 5 (très élevé) |
| **goout** | Fréquence des sorties avec des amis | 1 (très faible) à 5 (très élevé) |
| **Dalc** | Consommation d'alcool en semaine | 1 (très faible) à 5 (très élevé) |
| **Walc** | Consommation d'alcool le weekend | 1 (très faible) à 5 (très élevé) |
| **health** | État de santé actuel | 1 (très mauvais) à 5 (très bon) |
| **romantic** | Dans une relation amoureuse | Binaire (yes/no) |

### 3.6 Variables cibles (Notes)

| Variable | Description | Échelle |
|----------|-------------|---------|
| **G1** | Note du premier trimestre | 0 à 20 |
| **G2** | Note du deuxième trimestre | 0 à 20 |
| **G3** | Note finale (cible principale) | 0 à 20 |

**Note** : Le système de notation portugais utilise une échelle de 0 à 20, où 10 représente la note de passage minimum.

---

## 4. Analyse des Facteurs de Performance

### 4.1 Facteurs positifs influençant la réussite scolaire

#### Facteurs fortement corrélés au succès :
1. **Aspiration aux études supérieures** (higher = yes)
   - Les étudiants souhaitant poursuivre des études supérieures obtiennent des résultats significativement meilleurs

2. **Support familial éducatif** (famsup = yes)
   - L'implication parentale dans l'éducation améliore les performances

3. **Historique scolaire positif** (failures = 0)
   - L'absence d'échecs antérieurs est un excellent prédicteur de réussite future

4. **Temps d'étude adéquat** (studytime élevé)
   - Une corrélation positive existe entre le temps consacré aux études et les résultats

5. **Niveau d'éducation des parents** (Medu, Fedu élevés)
   - Un niveau d'éducation parental élevé est associé à de meilleures performances

6. **Accès à internet** (internet = yes)
   - Facilite l'accès aux ressources éducatives complémentaires

### 4.2 Facteurs négatifs impactant la performance

#### Facteurs de risque identifiés :
1. **Absentéisme** (absences élevées)
   - Forte corrélation négative entre le nombre d'absences et les résultats finaux

2. **Échecs scolaires antérieurs** (failures > 0)
   - Les échecs passés sont le prédicteur le plus fort d'échec futur

3. **Consommation d'alcool** (Dalc, Walc élevés)
   - Impact négatif sur la concentration et les résultats académiques

4. **Sorties sociales excessives** (goout élevé)
   - Lorsqu'elles empiètent sur le temps d'étude

5. **Temps de trajet important** (traveltime > 2)
   - Fatigue et temps réduit pour les études

6. **Relations amoureuses** (romantic = yes)
   - Peut détourner l'attention des études (corrélation faible mais mesurable)

### 4.3 Facteurs neutres ou à impact variable

- **Sexe de l'étudiant** : Impact variable selon la matière
- **Type de résidence** (urbain/rural) : Faible impact direct
- **Activités extrascolaires** : Peuvent être bénéfiques si équilibrées
- **État de santé** : Impact modéré et indirect

---

## 5. Applications Pratiques

### 5.1 Système d'alerte précoce
Le modèle permet d'identifier précocement les étudiants à risque en analysant :
- L'historique scolaire (G1, G2)
- Les absences répétées
- Les échecs antérieurs
- Le faible temps d'étude

**Bénéfice** : Intervention proactive avant l'échec final

### 5.2 Allocation optimale des ressources
Les établissements peuvent cibler leurs ressources vers :
- Les étudiants sans support familial
- Ceux ayant des échecs antérieurs
- Les étudiants avec un absentéisme élevé
- Les cas nécessitant un tutorat personnalisé

### 5.3 Politiques éducatives basées sur les données
Les décideurs peuvent :
- Développer des programmes de soutien parental
- Renforcer l'accompagnement des étudiants à risque
- Mettre en place des politiques anti-absentéisme
- Promouvoir l'équilibre vie scolaire/sociale

### 5.4 Personnalisation de l'enseignement
Les enseignants peuvent adapter leur approche selon :
- Le profil socio-économique de l'étudiant
- Les difficultés identifiées précocement
- Les besoins spécifiques de support

---

## 6. Méthodologie d'Analyse

### 6.1 Techniques de Data Mining utilisées
L'étude originale a employé plusieurs algorithmes d'apprentissage automatique :
- **Arbres de décision** (Decision Trees)
- **Réseaux de neurones** (Neural Networks)
- **Support Vector Machines** (SVM)
- **Régression multiple**

### 6.2 Mesures de performance
Les modèles ont été évalués selon :
- Le taux de précision (Accuracy)
- Le coefficient de corrélation
- La matrice de confusion
- Le RMSE (Root Mean Square Error)

### 6.3 Validation des résultats
- Validation croisée (cross-validation)
- Séparation train/test
- Analyse de la significativité statistique

---

## 7. Limites de l'Étude

### 7.1 Limites méthodologiques
- **Contexte géographique limité** : Données issues de deux écoles portugaises uniquement
- **Période temporelle** : Données collectées sur une seule année académique
- **Biais culturel** : Résultats potentiellement non généralisables à d'autres systèmes éducatifs

### 7.2 Variables non capturées
Facteurs non inclus dans le dataset :
- Qualité de l'enseignement
- Motivation intrinsèque de l'étudiant
- Relations avec les pairs (au-delà des sorties)
- Problèmes de santé mentale
- Événements familiaux traumatiques

### 7.3 Considérations éthiques
- Respect de la vie privée des étudiants
- Risque de stigmatisation des étudiants "à risque"
- Nécessité d'une utilisation responsable des prédictions

---

## 8. Conclusion

### 8.1 Principaux enseignements
Le Student Performance Dataset démontre que la réussite scolaire est un phénomène multifactoriel influencé par :
- Le contexte socio-économique familial
- Les comportements académiques de l'étudiant
- Le support éducatif disponible
- Les facteurs sociaux et comportementaux

### 8.2 Impact pour l'éducation
Cette recherche illustre le potentiel de l'Educational Data Mining pour :
- Améliorer les taux de réussite scolaire
- Optimiser l'allocation des ressources éducatives
- Développer des interventions personnalisées et efficaces
- Transformer les systèmes éducatifs vers une approche plus data-driven

### 8.3 Perspectives futures
Les recherches futures pourraient :
- Étendre le dataset à d'autres contextes culturels et géographiques
- Intégrer des variables psychologiques et motivationnelles
- Développer des modèles de deep learning plus sophistiqués
- Créer des systèmes d'intervention en temps réel

---

## 9. Références

### 9.1 Publication originale
Cortez, P., & Silva, A. (2008). Using Data Mining to Predict Secondary School Student Performance. In A. Brito and J. Teixeira (Eds.), *Proceedings of 5th Future Business Technology Conference (FUBUTEC 2008)* (pp. 5-12). Porto, Portugal.

### 9.2 Source des données
UCI Machine Learning Repository
- **URL** : https://archive.ics.uci.edu/ml/datasets/Student+Performance
- **DOI** : 10.24432/C5TG7T
- **Licence** : Creative Commons Attribution 4.0 International (CC BY 4.0)

### 9.3 Ressources complémentaires
- Dataset accessible via le package Python : `ucimlrepo`
- Code d'accès : `fetch_ucirepo(id=320)`

---

## Annexes

### Annexe A : Statistiques descriptives du dataset

**Distribution des notes (G3) :**
- Moyenne : ~11-12/20
- Médiane : ~11/20
- Écart-type : ~3-4 points
- Taux de réussite (G3 ≥ 10) : ~60-70%

**Profil démographique :**
- Âge moyen : 16-17 ans
- Ratio filles/garçons : environ équilibré
- Étudiants en zone urbaine : ~75%

### Annexe B : Code d'accès et exploration du dataset

#### B.1 Chargement des données

```python
from ucimlrepo import fetch_ucirepo 
  
# Fetch dataset 
student_performance = fetch_ucirepo(id=320) 
  
# Data (as pandas dataframes) 
X = student_performance.data.features 
y = student_performance.data.targets 
  
# Metadata 
print(student_performance.metadata) 
  
# Variable information 
print(student_performance.variables)
```

#### B.2 Exploration préliminaire des données

```python
import pandas as pd

# Combiner les features et les targets
df = pd.concat([X, y], axis=1)

# Afficher les premières lignes
print(df.head())

# Informations sur le dataset
print(df.info())

# Statistiques descriptives
print(df.describe())

# Vérifier les valeurs manquantes
print(df.isnull().sum())
```

#### B.3 Visualisation des performances

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution des notes finales (G3)
plt.figure(figsize=(10, 6))
sns.histplot(y['G3'], bins=20, kde=True)
plt.title('Distribution des Notes Finales (G3)')
plt.xlabel('Note (0-20)')
plt.ylabel('Fréquence')
plt.show()

# Corrélation entre les variables
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Matrice de Corrélation')
plt.show()
```
<img width="851" height="547" alt="image" src="https://github.com/user-attachments/assets/3180e64c-9d60-459b-9456-a3b3cba06cd3" />
<img width="973" height="893" alt="image" src="https://github.com/user-attachments/assets/cadf5739-c888-427e-838c-3c2bd5a533e4" />


---

**Document préparé pour rapport académique ou professionnel**
*Date : Novembre 2025*
