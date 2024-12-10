# 📚 Classification d'images MNIST avec un réseau de neurones en C/C++  

Ce projet vise à implémenter de zéro un réseau de neurones en C ou C++ pour classifier les chiffres manuscrits de la base de données MNIST.

## 🖼️ Présentation du jeu de données  

MNIST est une base de données bien connue en machine learning. Elle contient :  
- **60 000 images d'entraînement** et **10 000 images de test**,  
- Des **images en noir et blanc** de taille 28x28 pixels,  
- Chaque image représente un **chiffre de 0 à 9**.  

Le but est de développer un modèle capable de classer correctement ces images dans la catégorie correspondante.

---

## 🎯 Objectifs du projet  

1. **Comprendre la théorie**  
   - Appréhender les concepts de base : structure d'un réseau de neurones, différentiation automatique, descente de gradient, etc.  
   - Étudier les techniques d'entraînement et de validation.  

2. **Implémenter un réseau de neurones**  
   - Créer un réseau de neurones entièrement en **C ou C++**, sans utiliser de frameworks externes.  
   - Inclure les fonctions d'activation, la rétropropagation et le calcul des gradients.  

3. **Explorer des améliorations**  
   - Optimisation via la **parallélisation avec mini-batches**.  
   - Ajustement des hyperparamètres (taux d'apprentissage, taille des couches, etc.).  

---

## 🛠️ Technologies utilisées  

- **Langages** : C/C++  
- **Bibliothèques externes** : Aucune, tout est implémenté de zéro.  
- **Dépendances éventuelles** : Utilisation de fichiers pour lire les données MNIST en format binaire.  

---

## 🚀 Étapes du projet  

1. Préparation :  
   - Chargement des données MNIST au format binaire.  
   - Visualisation de quelques exemples pour validation.  

2. Implémentation :  
   - Création des structures de données pour le réseau de neurones.  
   - Développement des algorithmes d'entraînement (descente de gradient, propagation avant et arrière).  

3. Tests :  
   - Mesure de la précision sur l'ensemble de test.  
   - Ajustement et optimisation du réseau.  

4. Améliorations :  
   - Parallélisation de l'entraînement pour les mini-batches.  
   - Ajout d'autres optimisations (dropout, régularisation, etc.).

---

## 📈 Résultats attendus  

- Un modèle capable d'atteindre une précision acceptable (> 90 %) sur l'ensemble de test.  
- Une implémentation bien documentée et modulaire permettant des extensions ultérieures.  

---

## 📜 Licence  

Ce projet est sous licence [GLP](LICENSE).
