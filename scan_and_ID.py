from adaptXRD import spectrum_analysis, visualizer, quantifier
from adaptXRD import oracle
import sys
import os


if __name__ == '__main__':

    temp = 25 # Temperature during scan
    min_angle = 10.0 # Min two-theta on all scans
    starting_max = 60.0 # Max two-theta on first scan
    interval = 10.0 # How much to increase two-theta by each scan
    instrument = 'Bruker' # Type of diffractometer
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

    # Define diffractometer object
    diffrac = oracle.Diffractometer(instrument)

    # Get spectrum filename
    assert len(os.listdir('Spectra')) == 1, 'Too many spectra'
    spec_filename = os.listdir('Spectra')[0]

    # Run initial scan with low precision
    x, y = diffrac.execute_scan(min_angle, starting_max, 'Low', temp, spec_filename)

    # Write data to Spectra directory for analysis
    with open('Spectra/%s' % 'CurrentSpectrum.xy', 'w+') as f:
        for (xval, yval) in zip(x, y):
            f.write('%s %s\n' % (xval, yval))

    # Phase ID parameters
    max_phases = 4 # default: a maximum 4 phases in each mixture
    cutoff_intensity = 2.5 # default: ID all peaks with I >= 2.5% maximum spectrum intensity
    wavelength = 'CuKa' # default: spectra was measured using Cu K_alpha radiation
    max_angle = 120.0 # Upper bound on two-theta

    # Adaptive scanning parameters
    adaptive = True # Run adaptive scan & analysis
    parallel = False # Adaptive scanning cannot be run in parallel
    min_conf = 40.0 # If minimum confidence < 40%, run more scans
    cam_cutoff = 30.0 # Re-scan two-theta where CAM differences exceed 30%

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
        if '--conf_cutoff' in arg:
            conf_cutoff = float(arg.split('=')[1])
        if '--cam_cutoff' in arg:
            cam_cutoff = float(arg.split('=')[1])

    spectrum_names, predicted_phases, confidences = spectrum_analysis.main('Spectra', 'References', max_phases, cutoff_intensity, wavelength,
        min_angle, starting_max, max_angle, interval, parallel, adaptive, min_conf, cam_cutoff, temp, instrument)

    for (spectrum_fname, phase_set, confidence) in zip(spectrum_names, predicted_phases, confidences):

        if '--all' not in sys.argv: # By default: only include phases with a confidence > 20%
            all_phases = phase_set.split(' + ')
            all_probs = [float(val) for val in confidence]
            final_phases, final_confidence = [], []
            for (ph, cf) in zip(all_phases, all_probs):
                if cf >= 20.0:
                    final_phases.append(ph)
                    final_confidence.append(cf)

            print('Temperature: %s C' % temp)
            print('Predicted phases: %s' % final_phases)
            print('Confidence: %s' % final_confidence)

        else: # If --all is specified, print *all* suspected phases
            final_phases = phase_set.split(' + ')
            print('Temperature: %s C' % temp)
            print('Predicted phases: %s' % phase_set)
            print('Confidence: %s' % confidence)

        if ('--plot' in sys.argv) and (phase_set != 'None'):

            # Format predicted phases into a list of their CIF filenames
            final_phasenames = ['%s.cif' % phase for phase in final_phases]

            # Plot measured spectrum with line profiles of predicted phases
            visualizer.main('Spectra', spectrum_fname, final_phasenames, min_angle, max_angle, wavelength)

        if ('--weights' in sys.argv) and (phase_set != 'None'):

            # Format predicted phases into a list of their CIF filenames
            final_phasenames = ['%s.cif' % phase for phase in final_phases]

            # Get weight fractions
            weights = quantifier.main('Spectra', spectrum_fname, final_phasenames, min_angle, max_angle, wavelength)
            print('Weight fractions: %s' % weights)

