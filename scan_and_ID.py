from autoXRD import visualizer, quantifier
from adaptXRD import oracle
import numpy as np
import adaptXRD
import sys
import os


if __name__ == '__main__':

    max_phases = 4 # A maximum of 4 phases identified in each mixture
    cutoff_intensity = 5.0 # Identify all peaks with I >= 5% of the largest peak
    wavelength = 'CuKa' # Measurements use Cu K_alpha radiation
    temp = 25 # Temperature used during scan (useful for in situ measurements)
    min_angle = 10.0 # Lower bound on scan range (two-theta, degrees)
    start_max = 60.0 # Upper bound on initial range (10-60 degrees is a good starting point)
    final_max = 140.0 # Upper bound on final range (highest possible two-theta)
    interval = 10.0 # How much to increase two-theta range by each iteration
    min_window = 5.0 # Minimum range of angles that are scanned at each iteration
    min_conf = 10.0 # Minimum confidence (%) included in predictions
    target_conf = 80.0 # Perform measurements until confidence exceeds 80% for all phases
    cam_cutoff = 25.0 # Re-scan two-theta where CAM differences exceed 25%
    instrument = 'Bruker' # Type of diffractometer (others may include 'Aeris' or 'Post hoc')
    existing_file = None # Used for post hoc analysis (spectrum file already exists)
    init_step, final_step = 0.02, 0.1 # Step size (deg) and scan time per step (s) for initial scan
    init_time, final_time = 0.01, 0.2 # Step size (deg) and scan time per step (s) for final scan (resampling)

    for arg in sys.argv:
        if '--max_phases' in arg:
            max_phases = int(arg.split('=')[1])
        if '--cutoff_intensity' in arg:
            cutoff_intensity = int(float(arg.split('=')[1]))
        if '--wavelength' in arg:
            wavelength = float(arg.split('=')[1])
        if '--temp' in arg:
            temp = int(float(arg.split('=')[1]))
        if '--min_angle' in arg:
            min_angle = float(arg.split('=')[1])
        if '--start_max' in arg:
            start_max = float(arg.split('=')[1])
        if '--final_max' in arg:
            final_max = float(arg.split('=')[1])
        if '--interval' in arg:
            interval = float(arg.split('=')[1])
        if '--min_window' in arg:
            min_window = float(arg.split('=')[1])
        if '--min_conf' in arg:
            min_conf = float(arg.split('=')[1])
        if '--target_conf' in arg:
            target_conf = float(arg.split('=')[1])
        if '--cam_cutoff' in arg:
            cam_cutoff = float(arg.split('=')[1])
        if '--temp' in arg:
            temp = int(float(arg.split('=')[1]))
        if '--instrument' in arg:
            instrument = str(arg.split('=')[1])
        if '--existing_file' in arg:
            existing_file = str(arg.split('=')[1])
        if '--initial_step' in arg:
            init_step = float(arg.split('=')[1])
        if '--initial_time' in arg:
            init_time = float(arg.split('=')[1])
        if '--final_step' in arg:
            final_step = float(arg.split('=')[1])
        if '--final_time' in arg:
            final_time = float(arg.split('=')[1])

    # Define diffractometer object
    diffrac = oracle.Diffractometer(instrument)

    # Run a fast initial scan
    prec = 'Low' # Low precision
    x, y = diffrac.execute_scan(min_angle, start_max, prec, temp, existing_file,
        init_step, init_time, final_step, final_time)

    # Write initial scan data to file
    spectrum_fname = 'ScanData.xy'
    if existing_file != None:
        spectrum_fname = existing_file
    with open('Spectra/%s' % spectrum_fname, 'w+') as f:
        for (xval, yval) in zip(x, y):
            f.write('%s %s\n' % (xval, yval))

    # Perform phase identification and guide XRD measurements
    spec_dir, ref_dir = 'Spectra', 'References'
    adaptive_analyzer = adaptXRD.AdaptiveAnalysis(spec_dir, spectrum_fname, ref_dir, max_phases,
        cutoff_intensity, wavelength, min_angle, start_max, final_max, interval, min_conf, target_conf,
        cam_cutoff, temp, instrument, min_window, init_step, init_time, final_step, final_time)
    phases, confidences, scale_factors = adaptive_analyzer.main

    # Load final angle
    xrd = np.loadtxt('Spectra/%s' % spectrum_fname)
    x = xrd[:, 0]
    measured_max = max(x)

    # Inform user of temperature if not at RT
    if temp != 25:
        print('Temperature: %s C' % temp)

    # Print phases with high confidence
    if '--all' not in sys.argv:
        final_phases, final_confidence, final_heights = [], [], []
        for (ph, cf, ht) in zip(phases, confidences, scale_factors):
            if cf >= 25.0:
                final_phases.append(ph)
                final_confidence.append(cf)
                final_heights.append(ht)

        print('Predicted phases: %s' % final_phases)
        print('Confidence: %s' % final_confidence)

    # Or if --all is specified, print *all* suspected phases
    else:
        final_phases = phases.copy()
        final_heights = scale_factors.copy()
        print('Predicted phases: %s' % phase_set)
        print('Confidence: %s' % confidence)

    if ('--plot' in sys.argv) and (phases != 'None'):

        save = False
        if '--save' in sys.argv:
            save = True

        # Format predicted phases into a list of their CIF filenames
        final_phasenames = ['%s.cif' % phase for phase in final_phases]

        # Plot measured spectrum with line profiles of predicted phases
        visualizer.main('Spectra', spectrum_fname, final_phasenames, final_heights, min_angle, measured_max, wavelength, save)

    if ('--weights' in sys.argv) and (phases != 'None'):

        # Format predicted phases into a list of their CIF filenames
        final_phasenames = ['%s.cif' % phase for phase in final_phases]

        # Get weight fractions
        weights = quantifier.main('Spectra', spectrum_fname, final_phasenames, final_heights, min_angle, measured_max, wavelength)
        weights = [round(val, 2) for val in weights]
        print('Weight fractions: %s' % weights)
