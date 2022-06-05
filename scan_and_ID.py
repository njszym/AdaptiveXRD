from autoXRD import visualizer, quantifier
from adaptXRD import oracle
import adaptXRD
import sys
import os


if __name__ == '__main__':

    temp = 25 # Temperature during scan
    min_angle = 10.0 # Min two-theta on all scans
    starting_max = 60.0 # Max two-theta on first scan
    interval = 10.0 # How much to increase two-theta by each scan
    min_window = 5.0 # Minimum scan window
    instrument = 'Bruker' # Type of diffractometer
    existing_file = None # Used for post hoc analysis
    for arg in sys.argv:
        if '--temp' in arg:
            temp = int(float(arg.split('=')[1]))
        if '--min_angle' in arg:
            min_angle = float(arg.split('=')[1])
        if '--starting_max' in arg:
            starting_max = float(arg.split('=')[1])
        if '--interval' in arg:
            interval = float(arg.split('=')[1])
        if '--instrument' in arg:
            instrument = str(arg.split('=')[1])
        if '--existing_file' in arg:
            existing_file = str(arg.split('=')[1])

    # Define diffractometer object
    diffrac = oracle.Diffractometer(instrument)

    # Run initial scan
    prec = 'Low' # Low precision
    x, y = diffrac.execute_scan(min_angle, starting_max, prec, temp, existing_file)

    # Write data
    spectrum_fname = 'ScanData.xy'
    if existing_file != None:
        spectrum_fname = existing_file
    with open('Spectra/%s' % spectrum_fname, 'w+') as f:
        for (xval, yval) in zip(x, y):
            f.write('%s %s\n' % (xval, yval))

    # Phase ID parameters
    max_phases = 3 # default: a maximum 4 phases in each mixture
    cutoff_intensity = 10.0 # default: ID all peaks with I >= 10% maximum spectrum intensity
    wavelength = 'CuKa' # default: spectra was measured using Cu K_alpha radiation
    max_angle = 100.0 # Upper bound on two-theta
    min_conf = 10.0 # Minimum confidence (%) included in predictions
    target_conf = 80.0 # Perform measurements until confidence exceeds 80% for all phases

    # Adaptive scanning parameters
    adaptive = True # Run adaptive scan & analysis
    parallel = False # Adaptive scanning cannot be run in parallel
    cam_cutoff = 25.0 # Re-scan two-theta where CAM differences exceed 25%

    # User-specified args
    for arg in sys.argv:
        if '--max_phases' in arg:
            max_phases = int(arg.split('=')[1])
        if '--cutoff_intensity' in arg:
            cutoff_intensity = int(float(arg.split('=')[1]))
        if '--wavelength' in arg:
            wavelength = float(arg.split('=')[1])
        if '--max_angle' in arg:
            max_angle = float(arg.split('=')[1])
        if '--adaptive' in arg:
            if arg.split('=')[1] == 'True':
                adaptive = True
        if '--min_conf' in arg:
            min_conf = float(arg.split('=')[1])
        if '--target_conf' in arg:
            target_conf = float(arg.split('=')[1])
        if '--cam_cutoff' in arg:
            cam_cutoff = float(arg.split('=')[1])

    spec_dir, ref_dir = 'Spectra', 'References'
    adaptive_analyzer = adaptXRD.AdaptiveAnalysis(spec_dir, spectrum_fname, ref_dir, max_phases, cutoff_intensity, wavelength, min_angle,
        starting_max, max_angle, interval, min_conf, target_conf, cam_cutoff, temp, instrument, min_window)
    phases, confidences = adaptive_analyzer.main

    if temp != 25:
        print('Temperature: %s C' % temp)

    if '--all' not in sys.argv:

        # By default: only include phases with a confidence > 25%
        certain_phases, certain_confidences = [], []
        for (ph, cf) in zip(phases, confidences):
            if cf >= 25.0:
                certain_phases.append(ph)
                certain_confidences.append(cf)

        print('Predicted phases: %s' % certain_phases)
        print('Confidence: %s' % certain_confidences)

    else: # If --all is specified, print *all* suspected phases

        print('Predicted phases: %s' % phases)
        print('Confidence: %s' % confidences)

    if ('--plot' in sys.argv) and (phase_set != 'None'):

        # Format predicted phases into a list of their CIF filenames
        if '--all' not in sys.argv:
            final_phasenames = ['%s.cif' % phase for phase in certain_phases]
        else:
            final_phasenames = ['%s.cif' % phase for phase in phases]

        # Plot measured spectrum with line profiles of predicted phases
        visualizer.main('Spectra', spectrum_fname, final_phasenames, min_angle, max_angle, wavelength)

    if ('--weights' in sys.argv) and (phase_set != 'None'):

        # Format predicted phases into a list of their CIF filenames
        if '--all' not in sys.argv:
            final_phasenames = ['%s.cif' % phase for phase in certain_phases]
        else:
            final_phasenames = ['%s.cif' % phase for phase in phases]

        # Get weight fractions
        weights = quantifier.main('Spectra', spectrum_fname, final_phasenames, min_angle, max_angle, wavelength)
        print('Weight fractions: %s' % weights)


