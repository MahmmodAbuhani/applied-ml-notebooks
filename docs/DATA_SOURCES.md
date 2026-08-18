# Data Sources

## UCI Bank Marketing

- Dataset page: <https://archive.ics.uci.edu/dataset/222/bank+marketing>
- Runtime archive URL: <https://archive.ics.uci.edu/static/public/222/bank+marketing.zip>
- File used: `bank-additional/bank-additional-full.csv` inside `bank-additional.zip`.
- Runtime ZIP SHA-256: `e0bf5f5de5b846e2f18e9d90606637267d46dfa260e0f17bb12e605db5efbeb4`.
- License: CC BY 4.0 as listed by the UCI Machine Learning Repository.
- Citation: Moro, S., Rita, P., and Cortez, P. (2014). *Bank Marketing*. UCI Machine Learning Repository. DOI: `10.24432/C5K306`.
- Dataset description: Portuguese bank direct-marketing campaign records from 2008 to 2010 with mixed categorical and numeric client, contact, campaign, and macroeconomic fields.

The notebook loads the ZIP directly from UCI at runtime, verifies the archive hash, and does not commit the CSV. The `duration` column is deliberately excluded from the pre-call feature set because it is only known after the call ends and would leak outcome information.

The Bank Marketing notebook includes an order-based temporal stress test because UCI describes the rows as ordered by date from May 2008 through November 2010, but the CSV does not include a complete row-level timestamp. The stress test is therefore a stricter validation caveat, not a formal timestamped deployment validation.

## Palmer Penguins

- Source repository: <https://github.com/allisonhorst/palmerpenguins>
- CSV used by the end-to-end notebook and demo: <https://raw.githubusercontent.com/allisonhorst/palmerpenguins/8957207b78d6ccd1b4654a9dd9c9041b657478ab/inst/extdata/penguins.csv>
- Source commit pinned for reproducibility: `8957207b78d6ccd1b4654a9dd9c9041b657478ab`.
- CSV SHA-256 at the pinned source: `f204db2c753b0937caac3cb35258562c14f073e4bbc76be24b4c51ce22767a93`.
- License: CC0-1.0 in the source repository.
- Dataset description: penguin species, island, bill measurements, flipper length, body mass, sex, and year for Palmer Archipelago penguins.

The project uses the CSV directly from the pinned public source URL during execution instead of committing a local copy of the dataset.

## scikit-learn Built-In Datasets

The remaining notebooks use small datasets bundled with the pinned `scikit-learn` runtime:

- [Breast Cancer Wisconsin](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [Wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)
- [Handwritten Digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html)
- [Diabetes](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)

The linked loader pages document each dataset's source and characteristics. For the Wine notebook, the loader identifies the UCI Machine Learning Repository Wine dataset; the UCI dataset page lists CC BY 4.0 and requires attribution. The Diabetes notebook uses the scikit-learn runtime loader. The notebooks load packaged copies at runtime, raw data is not redistributed, and no separate dataset files are committed. The repository-owned CSV under `tests/fixtures/` is a small synthetic-format test fixture, not a public research dataset.
