from adaptXRD import oracle
from scipy.signal import find_peaks, filtfilt
from dtw import dtw, warp
import warnings
import random
from tqdm import tqdm
import pymatgen as mg
from pymatgen.analysis.diffraction import xrd
import cv2
from cv2_rolling_ball import subtract_background_rolling_ball
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.python.keras.backend import eager_learning_phase_scope
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate as ip
import numpy as np
import multiprocessing
from multiprocessing import Pool, Manager
from pymatgen.core import Structure
from fastdtw import fastdtw
from shutil import copyfile
import math
import time


class SpectrumAnalyzer(object):
    """
    Class used to process and classify xrd spectra.
    """

    def __init__(self, spectra_dir, spectrum_fname, max_phases, cutoff_intensity, model=None, wavelen='CuKa', reference_dir='References', min_angle=10.0, max_angle=80.0, adaptive=False):
        """
        Args:
            spectrum_fname: name of file containing the
                xrd spectrum (in xy format)
            reference_dir: path to directory containing the
                reference phases (CIF files)
            wavelen: wavelength used for diffraction (angstroms).
                Defaults to Cu K-alpha radiation (1.5406 angstroms).
        """

        self.spectra_dir = spectra_dir
        self.spectrum_fname = spectrum_fname
        self.ref_dir = reference_dir
        self.calculator = xrd.XRDCalculator()
        self.max_phases = max_phases
        self.cutoff = cutoff_intensity
        self.wavelen = wavelen
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.adaptive = adaptive
        self.model = model
        self.kdp = KerasDropoutPrediction(self.model)

    @property
    def reference_phases(self):
        refs = [fname for fname in os.listdir(self.ref_dir) if fname[0] != '.']
        return sorted(refs)

    @property
    def suspected_mixtures(self):
        """
        Returns:
            prediction_list: a list of all enumerated mixtures
            confidence_list: a list of probabilities associated with the above mixtures
        """

        spectrum = self.formatted_spectrum

        if self.adaptive:
            prediction_list, confidence_list, backup_list = self.enumerate_routes(spectrum)
            return prediction_list, confidence_list, backup_list

        else:
            prediction_list, confidence_list = self.enumerate_routes(spectrum)
            return prediction_list, confidence_list

    def convert_angle(self, angle):
        """
        Convert two-theta into Cu K-alpha radiation.
        """

        orig_theta = math.radians(angle/2.)

        orig_lambda = self.wavelen
        target_lambda = 1.5406 # Cu k-alpha
        ratio_lambda = target_lambda/orig_lambda

        asin_argument = ratio_lambda*math.sin(orig_theta)

        # Curtail two-theta range if needed to avoid domain errors
        if asin_argument <= 1:
            new_theta = math.degrees(math.asin(ratio_lambda*math.sin(orig_theta)))
            return 2*new_theta

    @property
    def formatted_spectrum(self):
        """
        Cleans up a measured spectrum and format it such that it
        is directly readable by the CNN.

        Args:
            spectrum_name: filename of the spectrum that is being considered
        Returns:
            ys: Processed XRD spectrum in 4501x1 form.
        """

        ## Load data
        data = np.loadtxt('%s/%s' % (self.spectra_dir, self.spectrum_fname))
        x = data[:, 0]
        y = data[:, 1]

        ## Convert to Cu K-alpha radiation if needed
        if str(self.wavelen) != 'CuKa':
            Cu_x, Cu_y = [], []
            for (two_thet, intens) in zip(x, y):
                scaled_x = self.convert_angle(two_thet)
                if scaled_x is not None:
                    Cu_x.append(scaled_x)
                    Cu_y.append(intens)
            x, y = Cu_x, Cu_y

        # Allow some tolerance (0.2 degrees) in the two-theta range
        if (min(x) > self.min_angle) and np.isclose(min(x), self.min_angle, atol=0.2):
            x = np.concatenate([np.array([self.min_angle]), x])
            y = np.concatenate([np.array([y[0]]), y])
        if (max(x) < self.max_angle) and np.isclose(max(x), self.max_angle, atol=0.2):
            x = np.concatenate([x, np.array([self.max_angle])])
            y = np.concatenate([y, np.array([y[-1]])])

        assert (min(x) <= self.min_angle) and (max(x) >= self.max_angle), """
               Measured spectrum (%s, %s) does not span the specified two-theta range!
               Either use a broader spectrum or change the two-theta range via
               the --min_angle and --max_angle arguments (%s, %s).""" % (min(x), max(x), self.min_angle, self.max_angle)

        ## Fit to 4,501 values as to be compatible with CNN
        f = ip.CubicSpline(x, y)
        xs = np.linspace(self.min_angle, self.max_angle, 4501)
        ys = f(xs)

        ## Smooth out noise
        ys = self.smooth_spectrum(ys)

        ## Map to integers in range 0 to 255 so cv2 can handle
        ys = [val - min(ys) for val in ys]
        ys = [255*(val/max(ys)) for val in ys]
        ys = [int(val) for val in ys]

        ## Perform baseline correction with cv2
        pixels = []
        for q in range(10):
            pixels.append(ys)
        pixels = np.array(pixels)
        img, background = subtract_background_rolling_ball(pixels, 800, light_background=False,
                                             use_paraboloid=True, do_presmooth=False)
        yb = np.array(background[0])
        ys = np.array(ys) - yb

        ## Normalize from 0 to 100
        ys = np.array(ys) - min(ys)
        ys = list(100*np.array(ys)/max(ys))

        return ys

    def smooth_spectrum(self, spectrum, n=10):
        """
        Process and remove noise from the spectrum.

        Args:
            spectrum: list of intensities as a function of 2-theta
            n: parameters used to control smooth. Larger n means greater smoothing.
                20 is typically a good number such that noise is reduced while
                still retaining minor diffraction peaks.
        Returns:
            smoothed_ys: processed spectrum after noise removal
        """

        # Smoothing parameters defined by n
        b = [1.0 / n] * n
        a = 1

        # Filter noise
        smoothed_ys = filtfilt(b, a, spectrum)

        return smoothed_ys

    def enumerate_routes(self, spectrum, indiv_conf=[], indiv_pred=[], indiv_backup=[], confidence_list=[], prediction_list=[], backup_list=[], is_first=True, normalization=1.0):
        """
        A branching algorithm designed to explore all suspected mixtures predicted by the CNN.
        For each mixture, the associated phases and probabilities are tabulated.

        Args:
            spectrum: a numpy array containing the measured spectrum that is to be classified
            kdp: a KerasDropoutPrediction model object
            reference_phases: a list of reference phase strings
            indiv_conf: list of probabilities associated with an individual mixture (one per branch)
            indiv_pred: list of predicted phases in an individual mixture (one per branch)
            confidence_list: a list of averaged probabilities associated with all suspected mixtures
            predictions_list: a list of the phases predicted in all suspected mixtures
            max_phases: the maximum number of phases considered for a single mixture.
                By default, this is set to handle  up tothree-phase patterns. The function is readily
                extended to handle arbitrary many phases. Caution, however, that the computational time
                required will scale exponentially with the number of phases.
            is_first: determines whether this is the first iteration for a given mixture. If it is,
                all global variables will be reset
        Returns:
            prediction_list: a list of all enumerated mixtures
            confidence_list: a list of probabilities associated with the above mixtures
        """

        # Make prediction and confidence lists global so they can be updated recursively
        # If this is the top-level of a new mixture (is_first), reset all variables
        if is_first:
            global updated_pred, updated_conf
            updated_pred, updated_conf = None, None
            prediction_list, confidence_list = [], []
            indiv_pred, indiv_conf = [], []
            if self.adaptive:
                global updated_backup
                updated_backup = None
                indiv_backup, backup_list = [], []

        prediction, num_phases, certanties = self.kdp.predict(spectrum)

        # If no phases are suspected
        if num_phases == 0:

            # If individual predictions have been updated recursively, use them for this iteration
            if 'updated_pred' in globals():
                if updated_pred != None:
                    indiv_pred, indiv_conf = updated_pred, updated_conf
                    updated_pred, updated_conf = None, None

            confidence_list.append(indiv_conf)
            prediction_list.append(indiv_pred)

        # Explore all phases with a non-trival probability
        for i in range(num_phases):

            # If individual predictions have been updated recursively, use them for this iteration
            if 'updated_pred' in globals():
                if updated_pred != None:
                    indiv_pred, indiv_conf = updated_pred, updated_conf
                    updated_pred, updated_conf = None, None
                    if self.adaptive:
                        indiv_backup = updated_backup
                        updated_backup = None

            phase_index = np.array(prediction).argsort()[-(i+1)]
            predicted_cmpd = self.reference_phases[phase_index]

            if self.adaptive:
                # If there exists two phases with high probabilities
                if num_phases > 1:
                    # For 1st most probable phase, choose 2nd most probable as backup
                    if i == 0:
                        backup_index = np.array(prediction).argsort()[-(i+2)]
                    # For 2nd most probable phase, choose 1st most probable as backup
                    # For 3rd most probable phase, choose 2nd most probable as backup (and so on)
                    elif i >= 1:
                        backup_index = np.array(prediction).argsort()[-i]
                    backup_cmpd = self.reference_phases[backup_index]
                # If only one phase is suspected, no backups are needed
                else:
                    backup_cmpd = None

            # If the predicted phase has already been identified for the mixture, ignore and move on
            if predicted_cmpd in indiv_pred:
                if i == (num_phases - 1):
                    confidence_list.append(indiv_conf)
                    prediction_list.append(indiv_pred)
                    updated_conf, updated_pred = indiv_conf[:-1], indiv_pred[:-1]
                    if self.adaptive:
                        backup_list.append(indiv_backup)
                        updated_backup = indiv_backup[:-1]
                continue

            # Otherwise if phase is new, add to the suspected mixture
            indiv_pred.append(predicted_cmpd)

            # Tabulate the probability associated with the predicted phase
            indiv_conf.append(certanties[i])

            # Tabulate alternative phases
            if self.adaptive:
                indiv_backup.append(backup_cmpd)

            # Subtract identified phase from the spectrum
            reduced_spectrum, norm = self.get_reduced_pattern(predicted_cmpd, spectrum, last_normalization=normalization)

            # If all phases have been identified, tabulate mixture and move on to next
            if norm == None:
                confidence_list.append(indiv_conf)
                prediction_list.append(indiv_pred)
                if self.adaptive:
                    backup_list.append(indiv_backup)
                if i == (num_phases - 1):
                    updated_conf, updated_pred = indiv_conf[:-2], indiv_pred[:-2]
                    if self.adaptive:
                        updated_backup = indiv_backup[:-2]
                else:
                    indiv_conf, indiv_pred = indiv_conf[:-1], indiv_pred[:-1]
                    if self.adaptive:
                        indiv_backup = indiv_backup[:-1]
                continue

            else:
                # If the maximum number of phases has been reached, tabulate mixture and move on to next
                if len(indiv_pred) == self.max_phases:
                    confidence_list.append(indiv_conf)
                    prediction_list.append(indiv_pred)
                    if self.adaptive:
                        backup_list.append(indiv_backup)
                    if i == (num_phases - 1):
                        updated_conf, updated_pred = indiv_conf[:-2], indiv_pred[:-2]
                        if self.adaptive:
                            updated_backup = indiv_backup[:-2]
                    else:
                        indiv_conf, indiv_pred = indiv_conf[:-1], indiv_pred[:-1]
                        if self.adaptive:
                            indiv_backup = indiv_backup[:-1]
                    continue

                # Otherwise if more phases are to be explored, recursively enter enumerate_routes with the newly reduced spectrum
                if self.adaptive:
                    prediction_list, confidence_list, backup_list = self.enumerate_routes(reduced_spectrum, indiv_conf=indiv_conf,
                        indiv_pred=indiv_pred, indiv_backup=indiv_backup, confidence_list=confidence_list,
                        prediction_list=prediction_list, backup_list=backup_list, is_first=False, normalization=norm)
                else:
                    prediction_list, confidence_list = self.enumerate_routes(reduced_spectrum, indiv_conf=indiv_conf,
                        indiv_pred=indiv_pred, confidence_list=confidence_list, prediction_list=prediction_list, is_first=False, normalization=norm)

        if self.adaptive:
            return prediction_list, confidence_list, backup_list
        else:
            return prediction_list, confidence_list

    def get_reduced_pattern(self, predicted_cmpd, orig_y, last_normalization=1.0):
        """
        Subtract a phase that has already been identified from a given XRD spectrum.
        If all phases have already been identified, halt the iteration.

        Args:
            predicted_cmpd: phase that has been identified
            orig_y: measured spectrum including the phase the above phase
            last_normalization: normalization factor used to scale the previously stripped
                spectrum to 100 (required by the CNN). This is necessary to determine the
                magnitudes of intensities relative to the initially measured pattern.
            cutoff: the % cutoff used to halt the phase ID iteration. If all intensities are
                below this value in terms of the originally measured maximum intensity, then
                the code assumes that all phases have been identified.
        Returns:
            stripped_y: new spectrum obtained by subtrating the peaks of the identified phase
            new_normalization: scaling factor used to ensure the maximum intensity is equal to 100
            Or
            If intensities fall below the cutoff, preserve orig_y and return Nonetype
                the for new_normalization constant.
        """

        # Simulate spectrum for predicted compounds
        pred_y = self.generate_pattern(predicted_cmpd)

        pred_y = np.array(pred_y)
        orig_y = np.array(orig_y)

        # Map pred_y onto orig_y through DTW
        distance, index_pairs = fastdtw(pred_y, orig_y, radius=50)
        warped_spectrum = orig_y.copy()
        for ind1, ind2 in index_pairs:
            distance = abs(ind1 - ind2)
            if distance <= 50:
                warped_spectrum[ind2] = pred_y[ind1]
            else:
                warped_spectrum[ind2] = 0.0
        warped_spectrum *= 100/max(warped_spectrum)

        # Scale warped spectrum so y-values match measured spectrum
        scaled_spectrum = self.scale_spectrum(warped_spectrum, orig_y)

        # Subtract scaled spectrum from measured spectrum
        stripped_y = self.strip_spectrum(scaled_spectrum, orig_y)
        stripped_y = self.smooth_spectrum(stripped_y)
        stripped_y = np.array(stripped_y) - min(stripped_y)

        # Normalization
        new_normalization = 100/max(stripped_y)
        actual_intensity = max(stripped_y)/last_normalization

        # If intensities remain above cutoff, return stripped spectrum
        if (new_normalization > 1.05) and (actual_intensity >= self.cutoff):
            stripped_y = new_normalization*stripped_y
            return stripped_y, last_normalization*new_normalization

        # Otherwise if intensities are too low, halt the enumaration
        else:
            return orig_y, None

    def calc_std_dev(self, two_theta, tau):
        """
        calculate standard deviation based on angle (two theta) and domain size (tau)
        Args:
            two_theta: angle in two theta space
            tau: domain size in nm
        Returns:
            standard deviation for gaussian kernel
        """
        ## Calculate FWHM based on the Scherrer equation
        K = 0.9 ## shape factor
        wavelength = self.calculator.wavelength * 0.1 ## angstrom to nm
        theta = np.radians(two_theta/2.) ## Bragg angle in radians
        beta = (K * wavelength) / (np.cos(theta) * tau) # in radians

        ## Convert FWHM to std deviation of gaussian
        sigma = np.sqrt(1/(2*np.log(2)))*0.5*np.degrees(beta)
        return sigma**2

    def generate_pattern(self, cmpd):
        """
        Calculate the XRD spectrum of a given compound.

        Args:
            cmpd: filename of the structure file to calculate the spectrum for
        Returns:
            all_I: list of intensities as a function of two-theta
        """

        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # don't print occupancy-related warnings
            struct = Structure.from_file('%s/%s' % (self.ref_dir, cmpd))
        equil_vol = struct.volume
        pattern = self.calculator.get_pattern(struct, two_theta_range=(self.min_angle, self.max_angle))
        angles = pattern.x
        intensities = pattern.y

        steps = np.linspace(self.min_angle, self.max_angle, 4501)

        signals = np.zeros([len(angles), steps.shape[0]])

        for i, ang in enumerate(angles):
            # Map angle to closest datapoint step
            idx = np.argmin(np.abs(ang-steps))
            signals[i,idx] = intensities[i]

        # Convolute every row with unique kernel
        # Iterate over rows; not vectorizable, changing kernel for every row
        domain_size = 25.0
        step_size = (self.max_angle - self.min_angle)/4501
        for i in range(signals.shape[0]):
            row = signals[i,:]
            ang = steps[np.argmax(row)]
            std_dev = self.calc_std_dev(ang, domain_size)
            # Gaussian kernel expects step size 1 -> adapt std_dev
            signals[i,:] = gaussian_filter1d(row, np.sqrt(std_dev)*1/step_size,
                                             mode='constant')

        # Combine signals
        signal = np.sum(signals, axis=0)

        # Normalize signal
        norm_signal = 100 * signal / max(signal)

        return norm_signal

    def scale_spectrum(self, pred_y, obs_y):
        """
        Scale the magnitude of a calculated spectrum associated with an identified
        phase so that its peaks match with those of the measured spectrum being classified.

        Args:
            pred_y: spectrum calculated from the identified phase after fitting
                has been performed along the x-axis using DTW
            obs_y: observed (experimental) spectrum containing all peaks
        Returns:
            scaled_spectrum: spectrum associated with the reference phase after scaling
                has been performed to match the peaks in the measured pattern.
        """

	# Ensure inputs are numpy arrays
        pred_y = np.array(pred_y)
        obs_y = np.array(obs_y)

        # Find scaling constant that minimizes MSE between pred_y and obs_y
        all_mse = []
        for scale_spectrum in np.linspace(1.1, 0.05, 101):
            ydiff = obs_y - (scale_spectrum*pred_y)
            mse = np.mean(ydiff**2)
            all_mse.append(mse)
        best_scale = np.linspace(1.0, 0.05, 101)[np.argmin(all_mse)]
        scaled_spectrum = best_scale*np.array(pred_y)

        return scaled_spectrum

    def strip_spectrum(self, warped_spectrum, orig_y):
        """
        Subtract one spectrum from another. Note that when subtraction produces
        negative intensities, those values are re-normalized to zero. This way,
        the CNN can handle the spectrum reliably.

        Args:
            warped_spectrum: spectrum associated with the identified phase
            orig_y: original (measured) spectrum
        Returns:
            fixed_y: resulting spectrum from the subtraction of warped_spectrum
                from orig_y
        """

        # Subtract predicted spectrum from measured spectrum
        stripped_y = orig_y - warped_spectrum

        # Normalize all negative values to 0.0
        fixed_y = []
        for val in stripped_y:
            if val < 0:
                fixed_y.append(0.0)
            else:
                fixed_y.append(val)

        return fixed_y

class KerasDropoutPrediction(object):
    """
    Ensemble model used to provide a probability distribution associated
    with suspected phases in a given xrd spectrum.
    """

    def __init__(self, model):
        """
        Args:
            model: trained convolutional neural network
                (tensorflow.keras Model object)
        """

        self.f = tf.keras.backend.function(model.layers[0].input, model.layers[-1].output)

    def predict(self, x, n_iter=250):
        """
        Args:
            x: xrd spectrum to be classified
        Returns:
            prediction: distribution of probabilities associated with reference phases
            len(certainties): number of phases with probabilities > 10%
            certanties: associated probabilities
        """

        x = [[val] for val in x]
        x = np.array([x])
        result = []
        with eager_learning_phase_scope(value=1):
            for _ in range(n_iter):
                result.append(self.f(x))

        result = np.array([list(np.array(sublist).flatten()) for sublist in result]) ## Individual predictions
        prediction = result.mean(axis=0) ## Average prediction

        all_preds = [np.argmax(pred) for pred in result] ## Individual max indices (associated with phases)

        counts = []
        for index in set(all_preds):
            counts.append(all_preds.count(index)) ## Tabulate how many times each prediction arises

        certanties = []
        for each_count in counts:
            conf = each_count/sum(counts)
            if conf >= 0.1: ## If prediction occurs at least 15% of the time
                certanties.append(conf)
        certanties = sorted(certanties, reverse=True)

        return prediction, len(certanties), certanties

class PhaseIdentifier(object):
    """
    Class used to identify phases from a given set of xrd spectra
    """

    def __init__(self, spectra_directory, reference_directory, max_phases, cutoff_intensity, wavelength, min_angle=10.0, starting_max=60.0, max_angle=80.0, interval=10.0, parallel=True, adaptive=False, min_conf=80.0, cam_cutoff=25.0, temp=25, instrument='Bruker'):
        """
        Args:
            spectra_dir: path to directory containing the xrd
                spectra to be analyzed
            reference_directory: path to directory containing
                the reference phases
        """

        self.num_cpu = multiprocessing.cpu_count()
        self.spectra_dir = spectra_directory
        self.ref_dir = reference_directory
        self.max_phases = max_phases
        self.cutoff = cutoff_intensity
        self.wavelen = wavelength
        self.parallel = parallel
        self.min_angle = min_angle
        self.starting_max = starting_max
        self.max_angle = max_angle
        self.interval = interval
        self.adaptive = adaptive
        self.min_conf = min_conf
        self.cam_cutoff = cam_cutoff
        self.temp = temp

        # Define diffractometer object
        self.diffrac = oracle.Diffractometer(instrument)

        # Adaptive scanning cannot be run in parallel
        if self.adaptive:
            self.parallel = False

    @property
    def all_predictions(self):
        """
        Returns:
            spectrum_names: filenames of spectra being classified
            predicted_phases: a list of the predicted phases in the mixture
            confidences: the associated confidence with the prediction above
        """

        reference_phases = sorted(os.listdir(self.ref_dir))
        spectrum_filenames = os.listdir(self.spectra_dir)

        if self.parallel:
            with Manager() as manager:
                pool = Pool(self.num_cpu)
                all_info = list(tqdm(pool.imap(self.classify_mixture, spectrum_filenames),
                    total=len(spectrum_filenames)))
                spectrum_fnames = [info[0] for info in all_info]
                predicted_phases = [info[1] for info in all_info]
                confidences = [info[2] for info in all_info]

        else:

            # Info contains filenames, predicted phases, and confidences
            all_info = []

            # Adaptive scan
            if self.adaptive:

                for filename in spectrum_filenames:

                    # Iteratively expand range in two-theta
                    halt = False
                    finely_sampled = []
                    angle_bounds = np.arange(self.starting_max, self.max_angle+0.1, self.interval)
                    for spec_max_angle in angle_bounds:

                        if halt:
                            continue

                        # Set up model
                        model_fname = 'Models/Model_%s.h5' % int(spec_max_angle)
                        self.model = tf.keras.models.load_model(model_fname, compile=False,
                            custom_objects={'sigmoid_cross_entropy_with_logits_v2': tf.nn.sigmoid_cross_entropy_with_logits})
                        final_conv_ind = 10 # change this value if the cnn architecture is modified
                        self.model.layers[final_conv_ind]._name = 'final_conv'

                        # If this is not the first scan, then we need to sample higher two-theta
                        if spec_max_angle != 60:
                            last_spectrum = np.loadtxt('Spectra/%s' % filename)
                            last_max = round(last_spectrum[:, 0][-1], 0)
                            last_max -= 1.0
                            scan_succeeded = self.increase_range(filename, last_max, spec_max_angle)
                            if not scan_succeeded:
                                continue

                        # Perform phase ID and check for backup phases
                        cmpds, probs, backups = self.classify_mixture(filename, max_angle=spec_max_angle, get_backups=True)

                        # If the minimum confidence is high, don't sample higher angles
                        if min(probs) > self.min_conf:
                            fnames = filename
                            phases = ' + '.join(cmpds)
                            confs = probs.copy()
                            halt = True
                            continue

                        uncert_x = []
                        x = np.linspace(self.min_angle, spec_max_angle, 4501)

                        # Re-sample based on phases with low confidence
                        for (cmpd, prob, backup) in zip(cmpds, probs, backups):
                            if (prob < self.min_conf) and (backup != None):
                                # First suspected phase
                                ph1_spec = generate_pattern(self.ref_dir, cmpd, spec_max_angle)
                                ph1_cam = gradcam_heatmap(ph1_spec, self.model, 'final_conv')
                                # Second suspected phase
                                ph2_spec = generate_pattern(self.ref_dir, backup, spec_max_angle)
                                ph2_cam = gradcam_heatmap(ph2_spec, self.model, 'final_conv')
                                # CAM difference between suspected phases
                                cam_diff = abs(np.array(ph1_cam) - np.array(ph2_cam))
                                # Identify two-theta where CAM difference exceed cutoff; these will be re-sampled to clarify
                                uncert_bounds = self.get_mismatch_range(cam_diff, bounds=(10, spec_max_angle), diff_cutoff=self.cam_cutoff)
                                for xrange in uncert_bounds:
                                    uncert_x += list(np.arange(min(xrange), max(xrange), 0.1))
                        uncert_x = sorted(list(set(uncert_x)))
                        uncert_x = [round(val, 1) for val in uncert_x]

                        # Don't re-sample regions that have already been sampled with high precision
                        uncert_x = sorted(list(set(uncert_x) - set(finely_sampled)))

                        # Re-sample any regions in two-theta where CAM difference is high
                        if len(uncert_x) > 0:
                            self.resample(filename, uncert_x)

                        # Keep track of xranges that have already been sampled with high precision
                        finely_sampled += uncert_x

                        # Re-run phase ID, now without considering backup phases
                        fnames, phases, confs = self.classify_mixture(filename, max_angle=spec_max_angle)

                        # If the minimum confidence is high, don't sample higher angles
                        if min(confs) > self.min_conf:
                            halt = True

                    all_info.append([fnames, phases, confs])


            # Non-adaptive (single) scan
            else:
                for filename in spectrum_filenames:
                    all_info.append(self.classify_mixture(filename))

            # Parse final info
            spectrum_fnames = [info[0] for info in all_info]
            predicted_phases = [info[1] for info in all_info]
            confidences = [info[2] for info in all_info]

        return spectrum_fnames, predicted_phases, confidences

    def get_unique_ranges(self, known_ranges, proposed_ranges, merge):

        # Don't re-sample regions that have already been sampled with high precision
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

    def classify_mixture(self, spectrum_fname, max_angle=80.0, get_backups=False):
        """
        Args:
            fname: filename string of the spectrum to be classified
        Returns:
            fname: filename, same as in Args
            predicted_set: string of compounds predicted by phase ID algo
            max_conf: confidence associated with the prediction
        """

        total_confidence, all_predictions = [], []
        tabulate_conf, predicted_cmpd_set = [], []

        # Load model
        model_fname = 'Models/Model_%s.h5' % int(max_angle)
        self.model = tf.keras.models.load_model(model_fname, compile=False)
        final_conv_ind = 10 # change this value if the cnn architecture is modified
        self.model.layers[final_conv_ind]._name = 'final_conv'

        if get_backups:

            # Run phase ID and get backup phases
            spec_analysis = SpectrumAnalyzer(self.spectra_dir, spectrum_fname, self.max_phases, self.cutoff,
                model=self.model, wavelen=self.wavelen, min_angle=self.min_angle, max_angle=max_angle, adaptive=True)
            cmpds, probs, backups = spec_analysis.suspected_mixtures

            # Organize info
            avg_conf = [np.mean(conf) for conf in probs]
            max_conf_ind = np.argmax(avg_conf)
            final_probs = [100*val for val in probs[max_conf_ind]]
            final_cmpds = cmpds[max_conf_ind]
            final_backups = backups[max_conf_ind]

            return final_cmpds, final_probs, final_backups

        else:

            # Run phase ID (no backups)
            spec_analysis = SpectrumAnalyzer(self.spectra_dir, spectrum_fname, self.max_phases, self.cutoff,
                model=self.model, wavelen=self.wavelen, min_angle=self.min_angle, max_angle=max_angle, adaptive=False)
            mixtures, confidences = spec_analysis.suspected_mixtures

            # If classification is non-trival, identify most probable mixture
            if len(confidences) > 0:
                avg_conf = [np.mean(conf) for conf in confidences]
                max_conf_ind = np.argmax(avg_conf)
                final_confidences = [100*val for val in confidences[max_conf_ind]]
                predicted_cmpds = [fname[:-4] for fname in mixtures[max_conf_ind]]
                predicted_set = ' + '.join(predicted_cmpds)

            # Otherwise, return Nontype
            else:
                predicted_set = 'None'
                final_confidences = [0.0]

            return [spectrum_fname, predicted_set, final_confidences]

        spec_analysis = SpectrumAnalyzer(self.spectra_dir, spectrum_fname, self.max_phases, self.cutoff,
            model=self.model, wavelen=self.wavelen, min_angle=self.min_angle, max_angle=max_angle, adaptive=False)
        mixtures, confidences = spec_analysis.suspected_mixtures

        # If classification is non-trival, identify most probable mixture
        if len(confidences) > 0:
            avg_conf = [np.mean(conf) for conf in confidences]
            max_conf_ind = np.argmax(avg_conf)
            final_confidences = [100*val for val in confidences[max_conf_ind]]
            predicted_cmpds = [fname[:-4] for fname in mixtures[max_conf_ind]]
            predicted_set = ' + '.join(predicted_cmpds)

        # Otherwise, return None
        else:
            final_confidences = [0.0]
            predicted_set = 'None'

        return [spectrum_fname, predicted_set, final_confidences]

    def resample(self, fname, xvals):

        # Re-format xvals into a set of bounds
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
            if max(subset) - min(subset) >= 1.0:
                bounds.append([min(subset), max(subset)])

        # Load the original spectrum
        data = np.loadtxt('%s/%s' % ('Spectra', fname))
        x_main = data[:, 0]
        y_main = data[:, 1]

        # Re-sample each xrange with high precision
        for min_max in bounds:
            min_angle = min_max[0]
            max_angle = min_max[1]

            # Get maximum intensity in sub-range
            orig_y = []
            for (xv, yv) in zip(x_main, y_main):
                if (xv >= min_angle) and (xv <= max_angle):
                    orig_y.append(yv)
            orig_max = max(orig_y)

            x_interp, y_interp = self.diffrac.execute_scan(min_angle, max_angle, 'High', self.temp, fname)

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

        # To avoid precision errors
        max_angle += 1

        # Load the original spectrum
        data = np.loadtxt('%s/%s' % ('Spectra', fname))
        x_main = data[:, 0]
        y_main = data[:, 1]

        # Sample higher two-theta and append to original range
        x_new, y_new = self.diffrac.execute_scan(min_angle, max_angle, 'Low', self.temp, fname)

        if None in x_new:
            return False

        else:

            x_main, y_main = self.splice_spectra(x_main, y_main, x_new, y_new)

            # Write to Spectra
            with open('%s/%s' % ('Spectra', fname), 'w+') as f:
                for (xv, yv) in zip(x_main, y_main):
                    f.write('%s %s\n' % (xv, yv))

            return True

    def splice_spectra(self, x_main, y_main, x_new, y_new):
        """
        Splice two sets of spectra together. Note that intensities are not necessarily
        continuous and smooth if count time changes between scans; hence, normalization
        will likely be necessary in experiment.

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
        Caution: should only be used with spice_spectra.
            You need to *splice* spectra, not just *merge* them.
            Otherwise, spectrum won't be smooth if x-values from
            different scans are overlapping with one another.
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
            if max(subrange) - min(subrange) >= 5.0:
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
    heatmap = [val - min(heatmap) for val in heatmap]
    heatmap = [100*val/max(heatmap) for val in heatmap]

    return heatmap

def calc_std_dev(two_theta, tau):
    """
    calculate standard deviation based on angle (two theta) and domain size (tau)
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



def main(spectra_directory, reference_directory, max_phases=3, cutoff_intensity=10, wavelength='CuKa', min_angle=10.0, starting_max=60.0, max_angle=80.0, interval=10.0, parallel=True, adaptive=False, min_conf=80.0, cam_cutoff=25.0, temp=25, instrument='Bruker'):

    phase_id = PhaseIdentifier(spectra_directory, reference_directory, max_phases, cutoff_intensity,
        wavelength, min_angle, starting_max, max_angle, interval, parallel, adaptive, min_conf, cam_cutoff, temp, instrument)

    spectrum_names, predicted_phases, confidences = phase_id.all_predictions

    return spectrum_names, predicted_phases, confidences
