# KABR Behaviour Dataset — V1 Execution

KABR is used as the initial behaviour-model dataset. The public KABR release contains eight behaviour labels: Walk, Trot, Run, Graze, Browse, Head Up, Auto-Groom, and Occluded. The KABR mini-scene release provides the numbered MP4 clips and matching CVAT XML action annotations.

## V1 strategy

Do not download the full KABR dataset or the original drone videos.

The downloader:
1. indexes the KABR mini-scene repository;
2. concurrently downloads only the small action XML annotations;
3. stops once the requested number of examples for every class has been found;
4. downloads only the selected mini-scene MP4/XML pairs;
5. writes data/raw/kabr/kabr_manifest.csv.

Pilot size: 3 clips per class = 24 clips.

## Execution

From the activated project virtual environment:

    git pull
    python scripts/download_kabr_subset.py --per-class 3 --workers 16

The expected terminal completion marker is:

    KABR_DOWNLOAD_OK

Then inspect:

    data/raw/kabr/kabr_manifest.csv

## V1 expansion

Only after the pilot is validated:

    python scripts/download_kabr_subset.py --per-class 20 --workers 16

Do not start with the 20-per-class run if the pilot has not been validated.

## Behaviour model

KABR labels remain the model labels. They are not silently converted into the project's operational risk ontology.

The ResNet18 + LSTM trainer reads the class vocabulary from the manifest/checkpoint, so it supports the eight KABR classes.

After training and evaluation, a separate conservative ontology mapping will connect KABR predictions to the operational risk engine. KABR does not contain a direct Chasing/Aggressive ontology equivalent for every project risk state, so those states must not be fabricated from KABR labels.

## Data provenance

Dataset: imageomics/KABR-mini-scene-raw-videos

The dataset is CC0-1.0. Cite the KABR WACV 2024 paper and dataset when publishing or presenting results.
