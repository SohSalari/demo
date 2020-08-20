import datetime 
from datetime import timedelta

import numpy as np

from stonesoup.types.detection import Detection
from stonesoup.types.state import WeightedGaussianState
from stonesoup.types.mixture import GaussianMixture


from stonesoup.types.detection import GaussianMixtureDetection
from stonesoup.reader import DetectionReader
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.types.array import StateVector, CovarianceMatrix


def detector():
    class TestDetector(DetectionReader):
        @BufferedGenerator.generator_method
        def detections_gen(self):
            time = datetime.datetime(2018, 1, 1, 14)
            for step in range(5):
                print("Step1:", step)
                yield time, {GaussianMixtureDetection([
                    WeightedGaussianState(([step, step]), np.diag([1, 1]), timestamp=time, weight=0.4),
                    WeightedGaussianState(([step, step]), np.diag([1, 1]), timestamp=time, weight=0.6)],
                    timestamp=time)}
                time += datetime.timedelta(minutes=1)
    return TestDetector()

detector1 = detector()

#print(type(detector1))
for mi in detector1:
    print(mi)

#mean_1 = StateVector([0, 0])
#mean_2 = StateVector([1, 2])
#covar_1 = CovarianceMatrix(np.diag([1, 1]))
#covar_2 = CovarianceMatrix(np.diag([2, 3]))
#
#soft_measurement = GaussianMixtureDetection([
#    WeightedGaussianState(mean_1, covar_1, weight=0.4),
#    WeightedGaussianState(mean_2, covar_2, weight=0.6)],
#    timestamp=1)
#print(soft_measurement)
