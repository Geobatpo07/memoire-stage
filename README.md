# Mémoire de Stage M1 — Modélisation Mathématiques Appliquée aux Cyclones

Ce dépôt contient le mémoire de stage de Master 1 réalisé du 14 avril au 14 juin 2025, sous la direction de **M. Maximilian HASLER** à l'Université des Antilles.  
Le sujet porte sur l'**étude d'équations différentielles ordinaires (EDO) et partielles (EDP) pour la modélisation de cyclones tropicaux**.

---

## Objectifs du stage

- Lire et analyser des modèles simplifiés d'écoulements atmosphériques.
- Étudier le modèle cyclonique en symétrie de révolution décrit dans le préprint de A.-Y. LeRoux (2014).
- Résoudre numériquement un système d'EDO pour obtenir les profils radiaux de vitesse et de pression.
- Produire des visualisations claires des résultats.
- Rédiger un mémoire scientifique structuré en LaTeX.

---

## Contenu du dépôt

| Fichier/Dossier    | Description                                                    |
|--------------------|----------------------------------------------------------------|
| `main.tex`         | Fichier principal du mémoire en LaTeX                          |
| `introduction.tex` | Introduction générale                                          |
| `theorie.tex`      | Théorie du modèle et rappels mathématiques                     |
| `resultats.tex`    | Présentation des résultats numériques                          |
| `conclusion.tex`   | Conclusion et perspectives                                     |
| `biblio.bib`       | Références bibliographiques (BibTeX)                           |
| `sections/`        | Fichiers .tex qui seront considérés comme le corps du document |
| `figures/`         | Figures générées par les simulations et autres images utiles   |
| `simulations/`     | Scripts Python pour la simulation/résolution numérique         |
| `out/`             | Fichiers rendus après la compilation                           |
| `README.md`        | Ce fichier de description                                      |

---

## Technologies utilisées

- **LaTeX** (rédaction scientifique)
- **Python 3** (calculs et visualisations)
- **Dataspell** (pour l'édition)

---

## Compilation du mémoire

Pour compiler l'ensemble du projet localement :

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Visualisation des résultats
Les profils de vitesse et de pression ont été générés à l'aide de scripts Python.
Les résultats sont sauvegardés dans le dossier `figures/` et intégrés automatiquement dans le document LaTeX.

---

## Mini To-Do List

- [x] Rédiger la page de garde
- [x] Écrire l'introduction (`introduction.tex`)
- [x] Décrire la théorie et le modèle (`theorie.tex`)
- [ ] Générer les résultats numériques (`resultats.tex`)
- [ ] Ajouter les figures de simulation
- [ ] Rédiger la conclusion (`conclusion.tex`)
- [x] Compléter la bibliographie (`biblio.bib`)
- [ ] Effectuer la relecture finale
- [ ] Effectuer les corrections après retour de l'encadrant
- [ ] Préparer la soutenance orale

---

## Tableau d'Avancement

| Tâche                            | Statut   | Commentaire                         |
|----------------------------------|----------|-------------------------------------|
| Recherche bibliographique        | En cours | Références principales collectées   |
| Rédaction Introduction           | Terminée |                                     |
| Développement Théorie            | En cours | Section détaillée                   |
| Résultats numériques             | À faire  | Profils générés en Python           |
| Rédaction Conclusion             | À faire  |                                     |
| Finalisation de la Bibliographie | En cours | À enrichir selon nouvelles lectures |
| Relecture et Corrections         | À faire  | Programmée fin mai                  |
| Préparation de la Soutenance     | À faire  | Programmée début juin               |

---

## Références principales
- A.-Y. LeRoux, Modélisation des écoulements dans leur environnement, preprint, 2014.
- Supports de cours de M. Hasler, CSIM-Hasler.
- Wikipedia, Force de Coriolis.
- Wikipedia, Équations d'Euler.
- Wikipedia, Équations de Navier-Stokes.
- Wikipedia, Équations de Barré de Saint-Venant.

La bibliographie complète est disponible dans le fichier biblio.bib.

---

## Auteur
Geovany Batista Polo LAGUERRE
Master 1 Mathématiques et Applications — Université des Antilles
Stage de recherche sous la direction de M. Maximilian HASLER.

---

## Licence
Ce dépôt est protégé par la licence [Creative Commons Attribution - NonCommercial - NoDerivatives 4.0 International (CC BY-NC-ND 4.0)](https://creativecommons.org/licenses/by-nc-nd/4.0/).

[![License: CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

Vous êtes libre de consulter, télécharger et partager ce mémoire **à des fins non commerciales**, **sans modifications**, et en **attribuant correctement l’auteur**.

© 2025 Geovany Batista Polo LAGUERRE – Université des Antilles