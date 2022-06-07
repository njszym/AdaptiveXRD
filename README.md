# Adaptive XRD

A packaged designed to integrate automated phase identification ([XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer)) with in-line guidance of XRD measurements.

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

By specifying these values during during training, several models will be created to represent each possible range of 2θ. For example:

* Model 1: trained on 2θ = [```min_angle```, ```start_max```]
* Model 2: trained on 2θ = [```min_angle```, ```start_max``` + ```interval```]
* Model 3: trained on 2θ = [```min_angle```, ```start_max``` + 2×```interval```]
* And so on...Until ```final_max``` is reached.

All trained models will be placed inside a folder called ```Models```. Their maximum angles will be specified in the filenames (e.g., ```Model_80.h5```).

## Performing measurements and analysis

Once a set of models are trained, they can be used to guide XRD scans and perform phase identification on the resulting patterns. This can be accomplished with the ```scan_and_ID``` script. For example:

```
python scan_and_ID.py --min_angle=10.0 --start_max=60.0 --final_max=140.0 --interval=10.0 --instrument='Aeris' --target_conf=80.0 --cam_cutoff=25.0 --min_window=5.0
```

The first few variables should match the values used during training. These dictate how the range 2θ will be expanded during the measurement.

The ```instrument``` variables is used to specify which diffractometer will be used. This is important as each instrument must be interfaced with in a unique way. Further details on this are given in the next section.

The later variables control resampling of 2θ within each range.

```target_conf``` represents the desired prediction confidence for each phase identified by [XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer). In cases where the prediction confidence falls below this value, additional measurements will be performed.

```cam_cutoff``` controls how much 2θ is scanned during each resampling iteration. A lower cutoff will lead to more resampling (and therefore a longer measurement time), whereas a higher cutoff leads to less resamlping (shorter scan time). A cutoff of 25% generally leads to a good balance between speed and accuracy.

```min_window``` defines the smallest range of 2θ that will be resampled. Generally, windows less than 5 degrees are inefficient as some time is required to set up the measurement, and such a small range will yield limited information.

Once the measurement and analysis are complete, an output will appear:

```
Predicted phases: (phase_1 + phase_2 + ...)
Confidence: (probabilities associated with the phases above)
```

And the measured spectrum can bound found in the ```Spectra``` folder.

To perfrom further analysis of this spectrum post hoc, the [XRD-AutoAnalyzer](https://github.com/njszym/XRD-AutoAnalyzer) can be used.

## Choice of diffractometer

As mentioned in the previous section, the diffractometer should be specified at runtime using the ```--instrument``` flag. This will tell adaptiveXRD how to interface with the diffractometer (see ```adaptXRD/oracle/__init__.py```).

Currently, this package supports three instrument types, described below.

### Aeris

Specifying ```--instrument="Aeris"``` enables an interface with a [Panalytical Aeris diffractometer](https://www.malvernpanalytical.com/en/products/product-range/aeris-range).

Communication is performed via TCP commands over an ethernet connection. Details regarding the Aeris communication system can be found in ```adaptXRD/AerisAI```. The IP address and results directory should be changed to reflect the user's setup.

### Bruker

Specifying ```--instrument="Bruker"``` enables an interface with a [Bruker D8 Advance diffractometer](https://www.bruker.com/en/products-and-solutions/diffractometers-and-scattering-systems/x-ray-diffractometers/d8-advance-family/d8-advance.html).

Communication is performed through a file handoff system, where input files with scan parameters are placed in a folder that is constantly monitored by the Bruker LabLims software. Experiment files (```.bsml```) and scan scripts (```.cs```) should be created by the user beforehand. The location of the job file should also be set before running any measurements on a new system. All information can be specified in the ```adaptXRD/oracle/__init__.py``` file.

### Post Hoc

Specifying ```--instrument="Post Hoc"``` enables interpolation of low- and high-precision spectra that are created beforehand. This is mostly used for testing purposes, as it represents a simulation of experimental sampling.

When using this flag, patterns with low and high resolution should be placed in folders named ```Low``` and ```High``` respectively.

Each filename should also be specified at runtime by using the ```--existing_file=FILENAME``` flag. As with other methods, the resulting spectrum will be placed in the ```Spectra``` folder.

### Additional diffractometers

AdaptiveXRD can be readily applied to new instrumentation, so long as an interface is accessible. In such cases, the ```adaptXRD/oracle/__init__.py``` should be modified accordingly.
