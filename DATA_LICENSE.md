# Data and model rights notice

This file is a **notice**, not a license grant. It explains which parts of this
repository are covered by the MIT License and which parts are subject to
third-party terms. It does not replace the authoritative IEEE-CIS/Kaggle rules.

## 1. Code license

The source code and documentation authored for this repository are licensed
under the MIT License. See [`LICENSE`](LICENSE).

Copyright (c) 2026 Mohammed M. A. Mahani.

## 2. The IEEE-CIS dataset is not included

The dataset used by this project is the **IEEE-CIS Fraud Detection** competition
dataset:

- Sponsor: IEEE Computational Intelligence Society (IEEE CIS).
- Data donated by: Vesta Corporation.
- Host: Kaggle.
- Competition: <https://www.kaggle.com/c/ieee-fraud-detection>
- Rules: <https://www.kaggle.com/c/ieee-fraud-detection/rules>
- Data page: <https://www.kaggle.com/c/ieee-fraud-detection/data>

The competition rules are the authoritative source of rights and restrictions.
Among other things, they state that Competition Data:

- may be accessed and used for **non-commercial purposes only**, including
  participating in the competition, using Kaggle forums, and academic research
  and education;
- must be protected with reasonable measures to prevent access by persons who
  have not agreed to the rules;
- must **not** be transmitted, duplicated, published, redistributed, or made
  available to any party not participating in the competition;
- may only be obtained by users who accept the rules on Kaggle.

The competition rules also contain provisions about publicly sharing code
developed in connection with the Competition Data. If those rules apply to you,
review them before redistributing this code or any derivative. This repository
does not include Competition Data.

This repository does **not** contain the raw or processed competition data,
the test files, out-of-fold probabilities, submission files, or any copy of the
dataset. The payload in `examples/sample_transaction.json` is synthetic and was
generated for this repository; it is not a row from the dataset.

Anyone who wants to reproduce the training pipeline must obtain the data from
Kaggle and accept the competition rules personally. The repository's
instructions describe how to use the data locally; they do not grant access to
it.

## 3. Model artifacts and derived outputs

The trained model and preprocessor in `models/` were fitted on the Competition
Data. They are provided for **research and education only**, "as is", without
warranty of any kind. They are not a redistribution of the dataset.

The MIT License applies to the source code and documentation. It does not grant
rights to the Competition Data and does not override the competition rules.
Because the model was trained on data subject to non-commercial terms, users are
responsible for determining whether their intended use is permitted. No
commercial rights to the data or to the trained model are granted here. For
commercial or production use, obtain independent legal advice and any required
permission from the rights holders.

Aggregate metrics, plots, SHAP summaries, and example outputs in `results/` are
derived outputs. They are provided for research and education and must not be
used to reconstruct the dataset. The example output file
`results/final_prediction_examples.csv` uses anonymized example labels in place
of dataset transaction identifiers; the model scores and SHAP factor names are
retained as research evidence.

## 4. Third-party software

Python dependencies are listed in `requirements.txt` and
`requirements-ci.txt`. They remain under their own licenses and are not
relicensed by this repository.

## 5. No legal advice

This notice records the project's understanding of the dataset terms at the time
of release preparation. It is not legal advice. If you intend to use the code,
model, or outputs commercially, or if you are unsure whether your use complies
with the competition rules, consult a qualified lawyer and the rights holders.
