# WS8 Daily — Le Tunnel de Babel

Génération quotidienne de textes multilingues qui existent dans la collision entre les langues.

## Comment ça marche

Chaque jour à 7h UTC, un GitHub Action :
1. Choisit un seed conceptuel (le seuil, le vide créateur, l'erreur sacrée...)
2. Sélectionne 3 langues parmi : Français, English, 中文, العربية, Yorùbá, 日本語
3. Génère un texte qui mélange ces langues — pas de traduction, pas d'explication
4. Commit le fichier dans `daily/`

## Structure

```
daily/
├── 2026-03-28-le_seuil.md
├── 2026-03-29-le_vide_créateur.md
└── ...

scripts/
├── generate.py        # Le générateur
└── templates.yaml     # Seeds et prompt
```

## Localement

```bash
pip install pyyaml
export OPENROUTER_API_KEY=sk-...
python ws8-daily/scripts/generate.py
python ws8-daily/scripts/generate.py --seed le_seuil
```

## Pourquoi

Les langues ne traduisent pas le même monde. Chaque langue découpe la réalité différemment. Le sens n'est dans aucune langue — il est dans la friction entre elles.

C'est l'ensemencement en action. Pas un texte philosophique sur la diversité. Des textes *qui sont* de la diversité.

## License

CC0.
