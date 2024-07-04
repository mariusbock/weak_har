
# Weak-Annotation of HAR Datasets using Vision Foundation Models

## Abstract
As wearable-based data annotation remains, to date, a tedious, time-consuming task requiring researchers to dedicate substantial time, benchmark datasets within the field of Human Activity Recognition in lack richness and size  compared to datasets available within related fields. Recently, vision foundation models such as CLIP have gained significant attention, helping the vision community advance in finding robust, generalizable feature representations. With the majority of researchers within the wearable community relying on vision modalities to overcome the limited expressiveness of wearable data and accurately label their to-be-released benchmark datasets offline, we propose a novel, clustering-based annotation pipeline to significantly reduce the amount of data that needs to be annotated by a human annotator. We show that using our approach, the annotation of centroid clips suffices to achieve average labelling accuracies close to 90% across three publicly available HAR benchmark datasets. Using the weakly annotated datasets, we further demonstrate that we can match the accuracy scores of fully-supervised deep learning classifiers across all three benchmark datasets.

## Supplementary Material
Additional results and figures can be found in the `supplementary_material.pdf`.

## Installation
Please follow instructions mentioned in the [INSTALL.md](/INSTALL.md) file.

## Download
The used datasets and extracted features can be downloaded [here](https://www.kaggle.com/datasets/anonymisedanon/iswc-2024-submission-3047). 

## Reproduce Feature Extraction
To reproduce the visual feature extraction, one needs to download each dataset's video data. Note that we used downsampled versions (12 FPS) of the video streams. The datasets can be downloaded here:
- WEAR: https://mariusbock.github.io/wear/index.html
- Wetlab: https://uni-siegen.sciebo.de/s/9h2aQbMaOIZsjc3
- ActionSense: https://action-sense.csail.mit.edu 

Note that the datset download mentioned above already contains all extracted feature embeddings.

## Reproduce Experiments
Once having installed requirements, one can rerun experiments. The `job_scripts` folder contains run-statements of all mentioned experiments.

To rerun the experiments without changing anything about the config files, please place the complete dataset download into a folder called `data` in the main directory of the repository.

## Seed averaging of experiments
To obtain a summarized overview of your seeded runs, run the `seed_averaging.py` script by changing the `path_to_preds` path-variable pointing towards a folder containing the logged experiments as separate folders, i.e.: 
- If you ran three experiments with varying different seeds place all three folders in a directory.
- Name each folder following the name structure `seed_X` where `X` is the employed seed of each experiment.

