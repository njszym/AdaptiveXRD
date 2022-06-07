# Adaptive XRD

A packaged designed to integrate automated phase identification ([XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer)) with in-line guidance of XRD measurements.

Work in progress.

## Installation

First clone the repository:

```
git clone https://github.com/njszym/XRD-AutoAnalyzer.git
```

Then, to install all required modules, navigate to the cloned directory and execute:

```
pip install . --user
```

Tensorflow version 2.5 is currently required. Later versions lead to overconfident models. We plan to resolve this in future updates.

## Training new models

Here, convolutional neural networks are trained using artifical XRD spectra that are generated from a given set of reference phases. This follows the methodology discussed in [XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer).

For adaptive XRD, multiple models are trained on distinct ranges of 2θ. To this end, models can be created as follows:

```
python construct_models.py --min_angle=10.0 --start_max=60.0 --final_max=140.0 --interval=10.0
```

Where ```start_max``` represents the upper bound on the first range of 2θ that will be sampled by the diffractometer. In cases where additional measurements are suggested, the scan range will be expanded by the amount specified (```interval```). This process continues until ```final_max``` is reached. Keep in mind that the maximum angle specified should be compatible with the diffractometer being used. For most instruments, 140 degrees represent an upper bound.

By specifying these values during during training, several models will be created to represent each possible range of 2θ (e.g., from ```min_angle``` to ```start_max```, from ```min_angle``` to ```start_max``` + ```interval```, and so on...until ```final_max``` is reached). 

