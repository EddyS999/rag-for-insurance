# RAG pour l'Assurance 🏦

Un système de **Retrieval-Augmented Generation (RAG)** spécialisé dans l'assurance, utilisant le modèle SmolLM2-360M et des embeddings pour répondre automatiquement aux questions sur les contrats d'assurance.

## 📋 Description

Ce projet implémente un système RAG complet qui permet de :
- Analyser et traiter des documents d'assurance (contrats, notices d'information)
- Répondre automatiquement aux questions des employés d'assurance
- Utiliser des embeddings sémantiques pour retrouver les informations pertinentes
- Générer des réponses contextuelles avec le modèle SmolLM2-360M

## 🚀 Fonctionnalités

- **Chunking intelligent** : Découpage des documents en paragraphes avec conservation du contexte
- **Embeddings sémantiques** : Utilisation de BAAI/bge-base-en-v1.5 pour la recherche vectorielle
- **Modèle de génération** : SmolLM2-360M-Instruct pour les réponses contextuelles
- **Évaluation** : Calcul du MRR (Mean Reciprocal Rank) pour mesurer la performance
- **Export des résultats** : Génération de fichiers CSV avec questions et réponses

## 📁 Structure du Projet

```
rag-for-insurance-master/
├── corpus/                    # Documents d'assurance
│   ├── AssurPlusVie.md       # Contrat AssurPlusVie
│   ├── epargne.md            # Contrat Épargne & Protection
│   ├── famille.md            # Contrat Famille
│   ├── patrimoine.md         # Contrat Patrimoine Vie Plus
│   ├── securite.md           # Contrat Sécurité Avenir
│   └── senior.md             # Contrat Capital Futur Senior
├── notebook.ipynb            # Notebook principal avec l'implémentation
├── questions.csv             # Questions de test avec réponses de référence
├── SmolLV2_360m.csv         # Résultats générés par le système RAG
└── weird_answers/            # Dossier pour les réponses problématiques
```

## 🛠️ Technologies Utilisées

- **Transformers** : Hugging Face pour les modèles de langage
- **FlagEmbedding** : Pour les embeddings sémantiques
- **Pandas** : Manipulation des données
- **NumPy** : Calculs numériques
- **SmolLM2-360M-Instruct** : Modèle de génération de texte
- **BAAI/bge-base-en-v1.5** : Modèle d'embeddings

## 📊 Corpus d'Assurance

Le projet inclut 6 types de contrats d'assurance :

1. **Patrimoine Vie Plus** - Contrat multisupport pour la protection et valorisation de l'épargne
2. **Sécurité Avenir** - Contrat multisupport avec garanties décès et vie
3. **Épargne & Protection** - Contrat avec versements flexibles
4. **Capital Futur Senior** - Contrat spécialisé pour les seniors
5. **Famille** - Contrat de protection familiale
6. **AssurPlusVie** - Contrat d'assurance vie complet

## 🔧 Installation

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/rag-for-insurance.git
cd rag-for-insurance
```

2. **Installer les dépendances**
```bash
pip install transformers torch pandas numpy FlagEmbedding
```

3. **Lancer le notebook**
```bash
jupyter notebook notebook.ipynb
```

## 💻 Utilisation

### 1. Chargement du modèle
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

smoll360m = "HuggingFaceTB/SmolLM2-360M-Instruct"
tokenizer = AutoTokenizer.from_pretrained(smoll360m)
model = AutoModelForCausalLM.from_pretrained(smoll360m)
```

### 2. Traitement des documents
```python
# Chunking des documents
chunks = []
for txt in texts:
    parsed_data = parse_document(txt)
    chunks.extend(parsed_data["chunks"])
```

### 3. Génération d'embeddings
```python
from FlagEmbedding import FlagModel

embedder = FlagModel('BAAI/bge-base-en-v1.5')
corpus_embedding = embedder.encode(chunks)
```

### 4. Recherche et génération de réponses
```python
# Recherche de contexte pertinent
context = get_context(query, chunks, corpus_embedding)

# Génération de réponse
messages = build_smoll_messages(query, chunks, corpus_embedding)
response = model.generate(messages)
```

## 📈 Évaluation

Le système utilise le **Mean Reciprocal Rank (MRR)** pour évaluer la qualité de la récupération d'informations :

```python
def compute_mrr(sim_score, acceptable_chunks):
    # Calcul du MRR pour mesurer la performance
    ranks = []
    for this_score, this_acceptable_chunks in zip(sim_score, acceptable_chunks):
        indexes = reversed(np.argsort(this_score))
        rank = 1 + next(i for i, idx in enumerate(indexes) if idx in this_acceptable_chunks)
        ranks.append(rank)
    
    return {
        "score": sum(1 / r if r < 6 else 0 for r in ranks) / len(ranks),
        "ranks": ranks,
    }
```

## 📝 Exemples de Questions

Le système peut répondre à des questions comme :
- "Quel est l'objet du contrat patrimoine vie plus ?"
- "Qui est l'assureur du contrat sécurité avenir ?"
- "Quelles sont les prestations du contrat sécurité avenir ?"
- "Quelles sont les modalités de versement du contrat Épargne & Protection ?"

## 🎯 Résultats

Les résultats sont exportés dans `SmolLV2_360m.csv` avec :
- Les questions originales
- Les réponses générées par le système RAG
- Les scores de similarité pour l'évaluation

## 🤝 Contribution

Les contributions sont les bienvenues ! Pour contribuer :

1. Fork le projet
2. Créer une branche pour votre fonctionnalité (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit vos changements (`git commit -am 'Ajout d'une nouvelle fonctionnalité'`)
4. Push vers la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 👥 Auteurs

- **Votre Nom** - *Développement initial* - [votre-github](https://github.com/votre-username)

## 🙏 Remerciements

- Hugging Face pour les modèles Transformers
- BAAI pour le modèle d'embeddings BGE
- La communauté open source pour les outils utilisés

---

**Note** : Ce projet est conçu pour un usage éducatif et de recherche. Pour un usage en production, veuillez adapter les paramètres selon vos besoins spécifiques.
