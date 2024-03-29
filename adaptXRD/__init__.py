from pymatgen.analysis.diffraction import xrd
from scipy.ndimage import gaussian_filter1d
from autoXRD import spectrum_analysis
from pymatgen.core import Structure
from scipy.signal import filtfilt
from adaptXRD import oracle
import tensorflow as tf
import numpy as np
import warnings
import os


class CustomDropout(tf.keras.layers.Layer):
    """
    Custom layer used to apply dropout in the CNN
    during training and inference.
    """

    def __init__(self, rate, **kwargs):
        super(CustomDropout, self).__init__(**kwargs)
        self.rate = rate

    def get_config(self):
        config = super().get_config()
        config.update({
            "rate": self.rate
        })
        return config

    # Always apply dropout
    def call(self, inputs, training=None):
        return tf.nn.dropout(inputs, rate=self.rate)


class AdaptiveAnalysis(object):
    """
    Main class used to run adaptive measurements
    and perform phase identification.
    """

    def __init__(self, spectrum_dir, spectrum_fname, reference_directory, max_phases,
        cutoff_intensity, wavelength, min_angle=10.0, start_max=60.0, final_max=100.0,
        interval=10.0, min_conf=10.0, target_conf=80.0, cam_cutoff=25.0, temp=25,
        instrument='Bruker', min_window=5.0, init_step=0.02, init_time=0.1,
        final_step=0.01, final_time=0.2):

        # Define spectrum path
        self.spectrum_dir = spectrum_dir
        self.spectrum_fname = spectrum_fname
        self.spectrum_path = os.path.join(self.spectrum_dir, self.spectrum_fname)

        # References used for training
        self.ref_dir = reference_directory
        self.reference_phases = sorted(os.listdir(self.ref_dir))

        # Parameters for autoXRD
        self.max_phases = max_phases
        self.cutoff = cutoff_intensity
        self.wavelen = wavelength
        self.min_angle = min_angle
        self.min_conf = min_conf

        # Parameters for adaptiveXRD
        self.start_max = start_max
        self.final_max = final_max
        self.target_conf = target_conf
        self.cam_cutoff = cam_cutoff
        self.min_window = min_window
        self.interval = interval
        self.temp = temp
        self.init_step = init_step
        self.init_time = init_time
        self.final_step = final_step
        self.final_time = final_time

        # Define diffractometer object
        self.diffrac = oracle.Diffractometer(instrument)

    @property
    def main(self):

        # Initialize adaptiveXRD variables
        finely_sampled = [] # Resampled parts of spectrum
        all_phases, all_confs, all_heights = [], [], [] # Ensemble of predictions, confidences, scale factors
        current_interval = self.interval
        angle_bounds = np.arange(self.start_max, self.final_max + 0.1, current_interval)

        # Iteratively expand scan range
        halt = False
        for max_angle in angle_bounds:

            # Check stopping criterion
            if halt:
                continue

            # Load trained CNN and set up model
            model_fname = 'Models/Model_%s.h5' % int(max_angle)
            self.model = tf.keras.models.load_model(model_fname, custom_objects={'CustomDropout': CustomDropout}, compile=False)
            final_conv_ind = 10 # Output of final conv layer. Change this value if CNN architecture is modified.
            self.model.layers[final_conv_ind]._name = 'final_conv'

            # If this is not the first scan, then sample higher two-theta
            if max_angle != self.start_max:
                last_max = max_angle - current_interval
                scan_succeeded = self.increase_range(self.spectrum_fname, last_max, max_angle)
                if scan_succeeded:
                    current_interval = self.interval
                else:
                    current_interval += self.interval
                    continue

            # Catch measurement errors
            if self.spectrum_fname not in os.listdir(self.spectrum_dir):
                halt = True
                continue

            # Exit loop if max angle is incorrect
            xrd = np.loadtxt(self.spectrum_path)
            x = xrd[:, 0]
            actual_max = max(x)
            if actual_max < max_angle:
                halt = True
                continue

            # Perform phase identification and check for backup phases
            spectrum_names, predicted_phases, confidences, backup_phases, scale_factors, reduced_spectra = spectrum_analysis.main(
                self.spectrum_dir, self.ref_dir, self.max_phases, self.cutoff, self.min_conf, self.wavelen, self.min_angle,
                max_angle, parallel=False, model_path=model_fname)
            cmpds, probs, backups, heights = predicted_phases[0], confidences[0], backup_phases[0], scale_factors[0]

            # Save for later
            prior_pred, prior_backup = cmpds.copy(), backups.copy()

            # Add current predictions to ensemble
            all_phases += cmpds.copy()
            all_confs += probs.copy()
            all_heights += heights.copy()

            # Calculate ensemble averaged predictions
            ensemble_phases, ensemble_confs, ensemble_heights = self.merge_predictions(all_phases, all_confs, all_heights)

            # If all confidences exceed cutoff, halt measurements
            if min(ensemble_confs) > self.target_conf:
                halt = True
                continue

            uncert_x = []
            x = np.linspace(self.min_angle, max_angle, 4501)

            # Re-sample based on phases with low confidence
            for (cmpd, prob, backup) in zip(cmpds, probs, backups):
                if (prob < self.target_conf) and (backup not in [None, 'None']):

                    # First suspected phase
                    ph1_spec = generate_pattern(self.ref_dir, cmpd, max_angle)
                    ph1_cam = gradcam_heatmap(ph1_spec, self.model, 'final_conv')

                    # Second suspected phase
                    ph2_spec = generate_pattern(self.ref_dir, backup, max_angle)
                    ph2_cam = gradcam_heatmap(ph2_spec, self.model, 'final_conv')

                    # CAM difference between suspected phases
                    cam_diff = abs(np.array(ph1_cam) - np.array(ph2_cam))

                    # Identify two-theta where CAM difference exceed cutoff; these will be re-sampled to clarify
                    uncert_bounds = self.get_mismatch_range(cam_diff, bounds=(10, max_angle), diff_cutoff=self.cam_cutoff)
                    for xrange in uncert_bounds:
                        uncert_x += list(np.arange(min(xrange), max(xrange), 0.1))

            uncert_x = sorted(list(set(uncert_x)))
            uncert_x = [round(val, 1) for val in uncert_x]

            # Don't resample areas that have already been sampled with high resolution
            uncert_x = sorted(list(set(uncert_x) - set(finely_sampled)))

            # Resample areas where CAM difference is high
            if len(np.array(uncert_x).flatten()) > 0:
                self.resample(self.spectrum_fname, uncert_x)

            # Keep track of two-theta ranges that have already been sampled with high precision
            finely_sampled += uncert_x

            # Redo phase identification and check for backup phases
            spectrum_names, predicted_phases, confidences, backup_phases, scale_factors, reduced_spectra = spectrum_analysis.main(
                self.spectrum_dir, self.ref_dir, self.max_phases, self.cutoff, self.min_conf, self.wavelen, self.min_angle,
                max_angle, parallel=False, model_path=model_fname)
            cmpds, probs, backups, heights = predicted_phases[0], confidences[0], backup_phases[0], scale_factors[0]

            # If the newly predicted phases are different from those suspected prior to resampling, raise a warning
            for new_ph in cmpds:
                if (new_ph not in prior_pred) and (new_ph not in prior_backup):
                    print('WARNING: %s is a new phase that was not detected before resampling' % new_ph)

            # Add new predictions to ensemble
            all_phases += cmpds.copy()
            all_confs += probs.copy()
            all_heights += heights.copy()

            # Calculate ensemble averaged predictions
            ensemble_phases, ensemble_confs, ensemble_heights = self.merge_predictions(all_phases, all_confs, all_heights)

            # If all confidences exceed cutoff, halt measurements
            if len(ensemble_confs) > 0:
                if min(ensemble_confs) > self.target_conf:
                    halt = True

        # Round off confidence and height values (two decimal points)
        ensemble_confs = [round(v, 2) for v in ensemble_confs]
        ensemble_heights = [round(v, 2) for v in ensemble_heights]

        return ensemble_phases, ensemble_confs, ensemble_heights

    def merge_predictions(self, preds, confs, heights):
        """
        Aggregate predictions through an ensemble approach
        whereby each phase is weighted by its confidence.
        """

        # Form dictionary for avg predictions
        avg_soln = {}
        for cmpd, cf, ht in zip(preds, confs, heights):
            if cmpd not in avg_soln.keys():
                avg_soln[cmpd] = [(cf, ht)]
            else:
                avg_soln[cmpd].append((cf, ht))

        # Iterate through each phase and record confidence, height
        unique_preds, avg_confs, avg_heights = [], [], []
        for cmpd in avg_soln.keys():
            unique_preds.append(cmpd)
            num_zeros = 2 - len(avg_soln[cmpd])
            avg_soln[cmpd] += [(0.0, 0.0)]*num_zeros
            avg_confs.append(np.mean([pair[0] for pair in avg_soln[cmpd]]))
            avg_heights.append(np.mean([pair[1] for pair in avg_soln[cmpd]]))

        # Sort from high to low confidence
        info = zip(unique_preds, avg_confs, avg_heights)
        info = sorted(info, key=lambda x: x[1])
        info.reverse()

        # Filter results into unique phases and exclude those with low confidence
        unique_cmpds, unique_confs, unique_heights = [], [], []
        for cmpd, cf, ht in info:
            if (len(unique_cmpds) < self.max_phases) and (cf > self.min_conf):
                unique_cmpds.append(cmpd)
                unique_confs.append(cf)
                unique_heights.append(ht)

        return unique_cmpds, unique_confs, unique_heights

    def get_unique_ranges(self, known_ranges, proposed_ranges, merge):
        """
        Get difference between proposed two-theta ranges
        and previously resampled ones.
        """
        for known_x in known_ranges:
            reframed_x = []
            for proposed_x in proposed_ranges:
                if (min(proposed_x) == min(known_x)) and (min(proposed_x) == min(known_x)):
                    pass
                elif (min(proposed_x) < min(known_x)) and (max(proposed_x) > min(known_x)) and (max(proposed_x) < max(known_x)):
                    reframed_x.append([min(proposed_x), min(known_x)])
                elif (min(proposed_x) < min(known_x)) and (max(proposed_x) > max(known_x)):
                    reframed_x.append([min(proposed_x), min(known_x)])
                    reframed_x.append([max(known_x), max(proposed_x)])
                elif (min(proposed_x) > min(known_x)) and (max(proposed_x) < max(known_x)):
                    pass
                elif (min(proposed_x) > min(known_x)) and (min(proposed_x) < max(known_x)) and (max(proposed_x) > max(known_x)):
                    reframed_x.append([max(known_x), max(proposed_x)])
                else:
                    reframed_x.append([min(proposed_x), max(proposed_x)])
            proposed_ranges = reframed_x.copy()

        if merge:
            return known_ranges + proposed_ranges
        else:
            return proposed_ranges

    def resample(self, fname, xvals):
        """
        Perform XRD measurements to resamples the
        range(s) of two-theta proposed in xvals.
        """

        # Reformat xvals into a set of bounds
        prev_val = 0.0
        xranges = []
        i = -1
        for val in xvals:
            if val - prev_val > 0.125:
                xranges.append([val])
                i += 1
            else:
                xranges[i].append(val)
            prev_val = val
        bounds = []
        for subset in xranges:
            if max(subset) - min(subset) >= self.min_window:
                bounds.append([min(subset), max(subset)])

        # Load the original spectrum
        data = np.loadtxt('%s/%s' % ('Spectra', fname))
        x_main = data[:, 0]
        y_main = data[:, 1]

        # Resample each xrange with high resolution
        for min_max in bounds:
            min_angle = min_max[0]
            max_angle = min_max[1]

            # Get maximum intensity in sub-range
            orig_y = []
            for (xv, yv) in zip(x_main, y_main):
                if (xv >= min_angle) and (xv <= max_angle):
                    orig_y.append(yv)
            orig_max = max(orig_y)

            x_interp, y_interp = self.diffrac.execute_scan(min_angle, max_angle, 'High', self.temp,
                fname, self.init_step, self.init_time, self.final_step, self.final_time)

            if None not in x_interp:

                # Normalize intensity by previous max
                y_interp = orig_max*np.array(y_interp)/max(y_interp)

                # Splice old and new spectra together
                x_main, y_main = self.splice_spectra(x_main, y_main, x_interp, y_interp)

        # Overwrite spectrum file with new data
        with open('%s/%s' % ('Spectra', fname), 'w+') as f:
            for (xval, yval) in zip(x_main, y_main):
                f.write('%s %s\n' % (xval, yval))

    def increase_range(self, fname, min_angle, max_angle):
        """
        Perform XRD measurement on two-theta
        from min_angle to max_angle.
        """

        # To avoid precision errors
        max_angle += 1

        # Load the original spectrum
        data = np.loadtxt('%s/%s' % ('Spectra', fname))
        x_main = data[:, 0]
        y_main = data[:, 1]

        # Sample higher two-theta and append to original range
        x_new, y_new = self.diffrac.execute_scan(min_angle, max_angle, 'Low', self.temp,
            fname, self.init_step, self.init_time, self.final_step, self.final_time)

        if None in x_new:
            return False

        else:

            # Splice spectra together
            x_main, y_main = self.splice_spectra(x_main, y_main, x_new, y_new)

            # Write to Spectra
            with open('%s/%s' % ('Spectra', fname), 'w+') as f:
                for (xv, yv) in zip(x_main, y_main):
                    f.write('%s %s\n' % (xv, yv))

            return True

    def splice_spectra(self, x_main, y_main, x_new, y_new):
        """
        Splice two sets of spectra together.

        Args:
            x_main, y_main: intial spectrum spanning full two-theta range
            x_new, y_new: subset of spectrum that has been re-sampled and is
                to be spliced with the initial spectrum
        Returns:
            x_final, y_final: spliced spectrum
        """

        x_main = np.array(x_main).flatten()
        y_main = np.array(y_main).flatten()

        spliced_x, spliced_y = [], []
        bounds = [min(x_new), max(x_new)]
        for (vx, vy)  in zip(x_main, y_main):
            if (vx < bounds[0]) or (vx > bounds[1]):
                spliced_x.append(vx)
                spliced_y.append(vy)

        final_x, final_y = self.merge_spectra(spliced_x, spliced_y, x_new, y_new)

        return final_x, final_y

    def merge_spectra(self, x1, y1, x2, y2):
        """
        Caution: should only be used with splice_spectra.
            You need to *splice* spectra, not just *merge* them.
            Otherwise, the spectrum won't be smooth if x-values
            from different scans overlap with one another.
        """

        x = list(x1) + list(x2)
        y = list(y1) + list(y2)
        sorted_spectrum = list(sorted(zip(x, y), key=lambda v: v[0]))
        final_x, final_y = [], []

        for index in range(len(sorted_spectrum) - 1):
            if not np.isclose(sorted_spectrum[index][0], sorted_spectrum[index+1][0], atol=0.002):
                final_x.append(sorted_spectrum[index][0])
                final_y.append(sorted_spectrum[index][1])

        if sorted_spectrum[-1][0] != final_x[-1]:
            final_x.append(sorted_spectrum[-1][0])
            final_y.append(sorted_spectrum[-1][1])

        return final_x, final_y

    def get_mismatch_range(self, cam_diff, bounds=(10, 80), diff_cutoff=10):
        """
        Identify regions of two-theta where measured and
        simulated CAMs differ significantly.
        """

        xrange = np.linspace(bounds[0], bounds[1], len(cam_diff))
        range_index = 0
        uncertain_x = [[]]
        for (x, y) in zip(xrange, cam_diff):
            if y > diff_cutoff:
                if uncertain_x[-1] != []:
                    # If xranges are < atol degrees apart, merge them into
                    # one larger range and sample in a single scan
                    if np.isclose(x, uncertain_x[range_index][-1], atol=5.0):
                        uncertain_x[range_index].append(x)
                    else:
                        uncertain_x.append([x])
                        range_index += 1
                else:
                    uncertain_x[range_index].append(x)

        bounds = []
        for subrange in uncertain_x:
            # Ignore narrow ranges
            if len(subrange) > 0:
                if max(subrange) - min(subrange) >= self.min_window:
                    bounds.append([round(min(subrange), 1), round(max(subrange), 1)])

        return bounds

def generate_pattern(ref_dir, cmpd, max_angle):
    """
    Calculate the XRD spectrum of a given compound.

    Args:
        cmpd: filename of the structure file to calculate the spectrum for
    Returns:
        all_I: list of intensities as a function of two-theta
    """

    if 'cif' not in cmpd:
        cmpd += '.cif'

    min_angle = 10.0 # Default
    calculator = xrd.XRDCalculator()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # don't print occupancy-related warnings
        struct = Structure.from_file('%s/%s' % (ref_dir, cmpd))
    equil_vol = struct.volume
    pattern = calculator.get_pattern(struct, two_theta_range=(min_angle, max_angle))
    angles = pattern.x
    intensities = pattern.y

    steps = np.linspace(min_angle, max_angle, 4501)

    signals = np.zeros([len(angles), steps.shape[0]])

    for i, ang in enumerate(angles):
        # Map angle to closest datapoint step
        idx = np.argmin(np.abs(ang-steps))
        signals[i,idx] = intensities[i]

    # Convolute every row with unique kernel
    # Iterate over rows; not vectorizable, changing kernel for every row
    domain_size = 25.0
    step_size = (max_angle - min_angle)/4501
    for i in range(signals.shape[0]):
        row = signals[i,:]
        ang = steps[np.argmax(row)]
        std_dev = calc_std_dev(ang, domain_size)
        # Gaussian kernel expects step size 1 -> adapt std_dev
        signals[i,:] = gaussian_filter1d(row, np.sqrt(std_dev)*1/step_size,
                                         mode='constant')

    # Combine signals
    signal = np.sum(signals, axis=0)

    # Normalize signal
    norm_signal = 100 * signal / max(signal)

    return norm_signal

def gradcam_heatmap(spectrum, model, last_conv_layer_name, pred_index=None):
    """
    Calculate the Grad-CAM for a given spectrum.
    Code adapted from https://keras.io/examples/vision/grad_cam/

    Args:
        spectrum: intensity array of shape (1, 4501, 1)
        model: trained model
        last_conv_layer_name: label of the final convolutional layer in the trained model
        pred_index: index of the class we want to build a CAM for. By default, this will
            be the predicted class for the given spectrum
    """

    # Format spectrum for processing
    spectrum = [[val] for val in spectrum]
    spectrum = np.array([spectrum])

    # Create a model that maps the input image to the activations
    # of the last conv layer and the output predictions
    grad_model = tf.keras.models.Model([model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output])

    # Compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(spectrum)
        if pred_index is None:
            pred_index = np.argmax(preds[0].numpy())
        else:
            pred_index = np.argsort(preds[0].numpy())[pred_index]
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=1)

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]

    # Flatten and make positive
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = abs(np.array(heatmap).flatten())

    # Smooth (filter noise)
    n = 10
    b = [1.0 / n] * n
    a = 1
    heatmap = filtfilt(b, a, heatmap)

    # Normalize from 0 to 100
    with warnings.catch_warnings():
        # Ignore errors related to small values
        warnings.simplefilter("ignore")
        heatmap = [val - min(heatmap) for val in heatmap]
        heatmap = [100*val/max(heatmap) for val in heatmap]

    return heatmap

def calc_std_dev(two_theta, tau):
    """
    Calculate standard deviation based on angle (two theta) and domain size (tau)
    Args:
        two_theta: angle in two theta space
        tau: domain size in nm
    Returns:
        standard deviation for gaussian kernel
    """
    ## Calculate FWHM based on the Scherrer equation
    K = 0.9 ## shape factor
    calculator = xrd.XRDCalculator()
    wavelength = calculator.wavelength * 0.1 ## angstrom to nm
    theta = np.radians(two_theta/2.) ## Bragg angle in radians
    beta = (K * wavelength) / (np.cos(theta) * tau) # in radians

    ## Convert FWHM to std deviation of gaussian
    sigma = np.sqrt(1/(2*np.log(2)))*0.5*np.degrees(beta)
    return sigma**2

