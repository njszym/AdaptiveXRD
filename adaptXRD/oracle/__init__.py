from adaptXRD.AerisAI import Aeris, write_spectrum
import numpy as np
import random
import time
import os


class Diffractometer:

    def __init__(self, instrument_name, results_dir='Results'):
        self.instrument_name = instrument_name
        self.results_dir = results_dir

        # You may add your own instrument here
        known_instruments = ['Bruker', 'Aeris', 'Post Hoc']
        assert self.instrument_name in known_instruments, 'Instrument is not known'

    def execute_scan(self, min_angle, max_angle, prec, temp, spec_fname=None):

        # High precision = slow scan
        # Low precision = fast scan
        assert prec in ['Low', 'High'], 'Precision must be High or Low, not %s' % prec

        # To avoid tolerance issues
        min_angle -= 0.1
        max_angle += 0.1

        """
        Each diffractometer requires unique interfacing scripts.

        Our current workflow is compatible with two diffractometers:
        - Bruker D8 Advance
        - Panalytical Aeris

        We also have a "Post Hoc" setting where measurements were
        performed beforehand and the adaptive algorithm learns
        to interpolate between high- and low-precision data.

        For use with a different instrument, add code here.
        """

        if self.instrument_name == 'Bruker':

            # Step size set by precision
            if prec == 'High':
                step_size = 0.01 # deg
                time_per_step = 0.2 # sec
            if prec == 'Low':
                step_size = 0.02 # deg
                time_per_step = 0.1 # sec

            # Expected measurement time
            expec_time = time_per_step*(max_angle - min_angle)/step_size

            # Allow some tolerance before calling timeout
            # e.g., it may take some time reach the measurement temperature
            tolerance = 600 # by default, allow 10 minutes

            # Write to params file; will be read by LabLims job file
            with open('ScanParams.txt', 'w+') as f:
                f.write('start_angle;end_angle;step_size;time_per_step;\n')
                f.write('%s;%s;%s;%s;' % (min_angle, max_angle, step_size, time_per_step))

            # Scan script and exp jobfile must be written beforehand!
            jobname = 'C:/ProgramData/Bruker AXS/LabLims/Job.xml'
            with open(jobname, 'w+') as f:
                f.write('<?xml version="1.0"?>\n')
                f.write('<MeasurementJob>\n')
                f.write('  <SamplePosition>1A01</SamplePosition>\n')
                f.write('  <Priority>50</Priority>\n')
                f.write('  <Permanent>false</Permanent>\n')
                f.write('  <ExperimentFileOrID>%sC.bsml</ExperimentFileOrID>\n' % temp)
                f.write('  <ScriptFile>FullScanScript.cs</ScriptFile>\n')
                f.write('</MeasurementJob>\n')

            # Wait until results file is detected
            done, failed = False, False
            start_time = time.time()
            timeout = expec_time + tolerance
            while not done:
                time.sleep(5) # Check folder once every 5 seconds
                result_files = os.listdir(self.results_dir)
                if len(result_files) > 0:
                    assert len(result_files) == 1, 'Too many result files'
                    fname = result_files[0]
                    done = True
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    done, failed = True, True

            # If measurement still failed, abort scan
            if failed:
                print('Measurement failed. No results file detected.')
                return [None], [None]

            # If scan was successful: load xy values
            x, y = [], []
            with open('Results/XRD.ascii') as f:
                for line in f.readlines()[3:]:
                    x.append(float(line.split(';')[2]))
                    y.append(float(line.split(';')[-1]))
            x = np.array(x)
            y = np.array(y)

            # Clean up results folder
            os.remove('Results/XRD.ascii')

            return x, y

        if self.instrument_name == 'Aeris':

            """
            Aeris scripts must be made manually by XRD MPcreator.
            Therefore, all possible programs should be created
            beforehand. By default, these should be named like:
                10-80_Low, 40-60_High, etc.
            Where the first two numbers represent the range
            that should be scanned in two-theta, and the
            subsequent word (Low or High) represents the
            scan precision.

            Here, we use scan windows as small as 5 deg.
            """

            window_size = max_angle - min_angle
            assert window_size >= 5.0, 'Scan window must be > 5 degrees'

            # Round to nearest 5
            min_angle = 5 * round(min_angle/5)
            max_angle = 5 * round(max_angle/5)

            # Program name
            program = '%s-%s_%s' % (min_angle, max_angle, prec)

            # Location on sample holder
            loc = 3

            # Choose a random sample ID
            sample_id = str(random.choice(range(1000)))

            aeris = Aeris()

            # Carry out scan and load spectrum
            success = aeris.scan(loc, sample_id, program)

            if success:
                x, y = aeris.load_xrdml(sample_id)

            else:
                assert False, 'Something went wrong with Aeris scan'

            return x, y


        if self.instrument_name == 'Post Hoc':

            """
            Spectra (xy) files should be placed in folders
            labaled "High" or "Low" to denote their precision.
            """

            # Load spectrum
            data = np.loadtxt('%s/%s' % (prec, spec_fname))
            full_x = data[:, 0]
            full_y = data[:, 1]

            # Get sub-range
            sub_x, sub_y = [], []
            for (xv, yv) in zip(full_x, full_y):
                if (xv >= min_angle) and (xv <= max_angle):
                    sub_x.append(xv)
                    sub_y.append(yv)
            sub_x = np.array(sub_x)
            sub_y = np.array(sub_y)

            return sub_x, sub_y

