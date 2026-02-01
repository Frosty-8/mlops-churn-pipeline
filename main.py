from gpt_folder_tree import create_tree, parse_tree, dry_run

gpt_ascii_text="""
mlops-churn-pipeline/
├── data/
│   ├── raw/
│   ├── processed/
│   └── live/
├── src/
│   ├── data/
│   │   └── generate_data.py
│   ├── features/
│   │   └── build_features.py
│   ├── training/
│   │   ├── train.py
│   │   └── retrain.py
│   ├── drift/
│   │   └── detect_drift.py
│   ├── api/
│   │   └── main.py
│   └── utils/
│       └── config.py
├── models/
│   └── best_model.pkl
├── .github/workflows/
│   └── retrain.yml
├── requirements.txt
├── README.md
└── start.sh
"""

tree = parse_tree(gpt_ascii_text)

dry_run(tree)

create_tree(tree)