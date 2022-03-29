Here you will find utilities for loading data and computing features from it.

## `make_features`
This provides the tools to load the Makusafe data, and to generate the features we use. All features are documented within `make_features/features.py`.

Instead of copying this to other directories to use, it is straightforward to add this directory to Python's path at runtime. The following code will do it in a Jupyter notebook within this git repo:

```python
import git
import os
import sys
root = git.Repo(os.getcwd(), search_parent_directories=True).git.rev_parse("--show-toplevel")
sys.path.append(os.path.join(root, 'data_processing'))
```

Then you can just import as if you were here, `import make_features`.

## process_other_datasets

Alas, the other data sets we have explored have large file sizes. They consist of:
 - SisFall (available from http://sistemic.udea.edu.co/en/investigacion/proyectos/english-falls/)
 - Erciyes (available from https://archive.ics.uci.edu/ml/datasets/Simulated+Falls+and+Daily+Living+Activities+Data+Set)
 - FallAllD (available from https://ieee-dataport.org/open-access/fallalld-comprehensive-dataset-human-falls-and-activities-daily-living)
 
All 3 are available from their original sources under a creative commons attribution licence. We have them downloaded on our shared server, with
their paths hardcoded in `process_other_datasets.py` for now. The output, which is a normalized form of each dataset converted to the structure of the Makusafe data set,
is available in `other_datasets`.