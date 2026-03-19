# Blind Assist Navigation System

This project presents a context-aware assistive navigation system designed for visually impaired individuals. The system integrates environment classification, domain-specific object detection, spatial reasoning, and decision-making to provide meaningful navigation guidance.

<p align="center">
  <img src="figures/architecture.png" width="700"/>
  <br>
  <em>Overall workflow of the proposed ultrasound image retrieval framework.</em>
</p>

  <em>Figure: Proposed dual-environment assistive navigation architecture</em>
</p>

## Features

* Dual-environment support (Indoor and Outdoor)
* Automatic environment classification
* Domain-specific object detection using YOLOv8
* Spatial reasoning based on object position and distance
* Priority-based obstacle selection
* Context-aware navigation decisions

## Datasets

* SUN RGB-D (Indoor)
* KITTI (Outdoor)

## Project Structure

```
Blind-Assist-System/
│
├── data_preparation/
├── training/
├── evaluation/
├── navigation_system/
├── configs/
├── figures/
├── results/
├── requirements.txt
└── README.md
```

## Installation

```
pip install -r requirements.txt
```

## Dataset Preparation

```
python data_preparation/download_datasets.py
```

## Training

```
python training/train_indoor.py
python training/train_outdoor.py
```

## Evaluation

```
python evaluation/evaluate.py
```

## Navigation System

```
python navigation_system/navigation.py
```

Ensure trained weights are placed in the `weights/` directory:

* indoor_best.pt
* outdoor_best.pt

## Method Overview

1. Image preprocessing
2. Environment classification
3. Model selection (Indoor / Outdoor)
4. Object detection
5. Spatial reasoning
6. Priority object selection
7. Decision generation

## Output

The system generates navigation feedback based on object proximity and environmental context.

## Requirements

* Python 3.x
* GPU recommended for training

## License

This project is intended for academic and research purposes.
