import numpy as np
import xmltodict
import socket
import time
import os


class Aeris:

    def __init__(self, ip='128.3.19.238', port=702, results_dir='/Users/Cederexp/Documents/SharedFolder', working_dir='./Results'):
        self.ip = ip
        self.port = port
        self.results_dir = results_dir
        self.working_dir = working_dir

    # Check if sample holder is occupied
    def check_status(self, loc):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.ip, self.port))
            mssg = bytes("@STATUS_REQUEST@LOCATION=0,%s@END" % loc, encoding="utf-8")
            s.sendall(mssg)
            data = s.recv(1024)
            print(repr(data))
            time.sleep(5)
            for output in str(data).split('@'):
                if 'STATE' in output:
                    status = output.split('=')[1]
            return status

    def scan(self, loc, sample_id='unknown_sample', program='10-140_2-min'):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.ip, self.port))
            mssg = bytes("@SAMPLE@ADD@SAMPLE_ID=%s@APPLICATION=%s@AT=0,%s@MEASURE=yes@END" % (sample_id, program, loc), encoding="utf-8")
            s.sendall(mssg)
            data = s.recv(1024)
            print(repr(data))
            time.sleep(5)
            filename = '%s.xrdml' % sample_id
            while filename not in os.listdir(self.results_dir):
                time.sleep(5)
            source = os.path.join(self.results_dir, filename)
            target = os.path.join(self.working_dir, filename)
            os.rename(source, target)
        time.sleep(5)
        print('Remove sample')
        self.remove(sample_id)
        print('Sample removed')

    # Necessary after each scan
    def remove(self, sample_id):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.ip, self.port))
            mssg = bytes("@SAMPLE@REMOVE@SAMPLE_ID=%s@END" % sample_id, encoding="utf-8")
            s.sendall(mssg)
            data = s.recv(1024)
            print(repr(data))
            time.sleep(5)

    def load_xrdml(self, sample_id):
        xrd_path = '%s/%s.xrdml' % (self.working_dir, sample_id)
        with open(xrd_path, 'r', encoding='utf-8') as file:
            xml_data = file.read()
        xrd_dict = xmltodict.parse(xml_data)

        min_angle = float(xrd_dict["xrdMeasurements"]["xrdMeasurement"]["scan"]["dataPoints"]["positions"][0]['startPosition'])
        max_angle = float(xrd_dict["xrdMeasurements"]["xrdMeasurement"]["scan"]["dataPoints"]["positions"][0]['endPosition'])

        intensities = xrd_dict["xrdMeasurements"]["xrdMeasurement"]["scan"]["dataPoints"]["counts"]["#text"]
        intensities = [float(val) for val in intensities.split()]
        angles = np.linspace(min_angle, max_angle, len(intensities))

        return angles, intensities

    def move(initial, target, sample_id='unknown_sample'):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((self.ip, self.port))
            mssg = bytes("@SAMPLE@MOVE@SAMPLE_ID=%s@AT=0,%s@TO=0,%s@END" % (sample_id, initial, target), encoding="utf-8")
            s.sendall(mssg)
            data = s.recv(1024)
            print(repr(data))
            time.sleep(5)


def write_spectrum(dir, sample_id, angles, intensities):
    filepath = os.path.join(dir, '%s.xy' % sample_id)
    with open(filepath, 'w+') as spec_file:
        for x, y in zip(angles, intensities):
            spec_file.write('%s %s\n' % (x, y))


