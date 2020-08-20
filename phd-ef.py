import datetime
from datetime import timedelta
import numpy as np

"------------------------------ Transition Model -----------------------------"
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel
from stonesoup.models.transition.linear import ConstantVelocity

transition_model = CombinedLinearGaussianTransitionModel((ConstantVelocity(0.3),
                                                          ConstantVelocity(0.3)))
#print("=========== Transition matrix: =============")
#print(transition_model.matrix(time_interval=timedelta(seconds=1)))   
#print("=========== Process Noise Covariance: =============")
#print(transition_model.covar(time_interval=timedelta(seconds=1)))   
                                            
"---------------------------- Measurement Model ------------------------------"
from stonesoup.models.measurement.linear import LinearGaussian

measurement_model = LinearGaussian(ndim_state=4, mapping=[0, 2], noise_covar=np.diag([1, 1]))

##print("=========== Measurement Matrix: =============")
#print(measurement_model.matrix())   
##print("=========== Measurement Model Covariance: =============")
#print(measurement_model.covar())   

#sensor_x = 10
#sensor_y = 0
#from stonesoup.models.measurement.nonlinear import CartesianToBearingRangeRate
#measurement_model = CartesianToBearingRangeRate(
#    4, # Number of state dimensions (position and velocity in 2D)
#    (0, 2), # Mapping measurement vector index to state index
#    np.diag([np.radians(0.1), 0.1]),  # Covariance matrix for Gaussian PDF
#    translation_offset=np.array([[sensor_x], [sensor_y]]) # Location of sensor in cartesian.
#)
#print(measurement_model.covar())   
#print(measurement_model.function([1,1,1]))   
#
#
#return

"-------------------- Multi-Target Ground Truth Simulator --------------------"
from stonesoup.simulator.simple import MultiTargetGroundTruthSimulator
from stonesoup.types.state import GaussianState
from stonesoup.types.array import StateVector, CovarianceMatrix
from stonesoup.types.numeric import Probability

groundtruth_sim = MultiTargetGroundTruthSimulator(
        transition_model = transition_model,
        initial_state = GaussianState(
                StateVector([[0], [0], [0], [0]]),
                CovarianceMatrix(np.diag([1000, 2, 1000, 2]))),
        timestep = datetime.timedelta(minutes=1),
        number_steps=3,  # Change the number to change the number of steps
        birth_rate=1,
        death_probability=Probability(0.05))

#for x in groundtruth_sim:
#    print(x)

"------------------ Detection Simulator (Creates Clutter) --------------------"
from stonesoup.simulator.simple import SimpleDetectionSimulator

detection_sim = SimpleDetectionSimulator(
        groundtruth=groundtruth_sim,
        measurement_model=measurement_model,
        meas_range=np.array([[-1, 1], [-1, 1]])*2000,  # Area to generate clutter
        detection_probability=Probability(0.9),
        clutter_rate=1.2)

#for x in detection_sim:
#    print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
#    print(x)
#    print("lenght:", len(x))

from stonesoup.types.detection import GaussianMixtureDetection
from stonesoup.reader import DetectionReader
from stonesoup.buffered_generator import BufferedGenerator
from stonesoup.types.state import WeightedGaussianState

def detector():
    class TestDetector(DetectionReader):
        @BufferedGenerator.generator_method
        def detections_gen(self):
            time = datetime.datetime(2020, 1, 1, 14)
            for step in range(3):
                print("Step1:", step)
                yield time, {GaussianMixtureDetection([
                    TaggedWeightedGaussianState(([step, step]), np.diag([2, 2]), timestamp=time, weight=0.3, tag=1),
                    TaggedWeightedGaussianState(([step, step]), np.diag([3, 3]), timestamp=time, weight=0.1, tag=1),
                    TaggedWeightedGaussianState(([step, step]), np.diag([4, 4]), timestamp=time, weight=0.2, tag=1),
                    TaggedWeightedGaussianState(([step, step]), np.diag([5, 5]), timestamp=time, weight=0.4, tag=1)],
                    timestamp=time), GaussianMixtureDetection([
                    TaggedWeightedGaussianState(([step, step]), np.diag([6, 6]), timestamp=time, weight=0.5, tag=2),
                    TaggedWeightedGaussianState(([step, step]), np.diag([7, 7]), timestamp=time, weight=0.3, tag=2),
                    TaggedWeightedGaussianState(([step, step]), np.diag([8, 8]), timestamp=time, weight=0.2, tag=2)],
                    timestamp=time), GaussianMixtureDetection([
                    TaggedWeightedGaussianState(([step, step]), np.diag([9, 9]), timestamp=time, weight=0.8, tag=3),
                    TaggedWeightedGaussianState(([step, step]), np.diag([9, 9]), timestamp=time, weight=0.2, tag=3)],
                    timestamp=time)}
                time += datetime.timedelta(seconds=1)
    return TestDetector()

detector1 = detector()
#print(detector1)
#for mi in detector1:
#    print(mi)

"----------------------------- Kalman Predictor ------------------------------"
from stonesoup.predictor.kalman import KalmanPredictor

kalman_predictor = KalmanPredictor(transition_model)

"------------------------------- Kalman Updater ------------------------------"
from stonesoup.updater.kalman import KalmanUpdater, ExtendedKalmanUpdater, UnscentedKalmanUpdater

kalman_updater = KalmanUpdater(measurement_model)

"--------------------------------- PHD Updater -------------------------------"
#from stonesoup.updater.softprocess import PHDUpdater
from stonesoup.updater.pointprocess import SOFTUpdater

updater = SOFTUpdater( 
    kalman_updater,
    clutter_spatial_density=detection_sim.clutter_spatial_density,
    prob_detection=detection_sim.detection_probability,
    prob_survival=1-groundtruth_sim.death_probability)

"-------------------------------- Hypothesiser -------------------------------"
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser

base_hypothesiser = DistanceHypothesiser(kalman_predictor, kalman_updater, Mahalanobis(), missed_distance=3)
hypothesiser = GaussianMixtureHypothesiser(base_hypothesiser, order_by_detection=True)

"---------------------------------- Initiator --------------------------------"
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.types.numeric import Probability

birth_component = TaggedWeightedGaussianState(
        groundtruth_sim.initial_state.state_vector,
        groundtruth_sim.initial_state.covar**2,
        weight=Probability(groundtruth_sim.birth_rate),
        tag="birth")

"""print(groundtruth_sim.initial_state.state_vector,)
print(groundtruth_sim.initial_state.covar**2,)
print(Probability(groundtruth_sim.birth_rate))
birth_mean = np.array([[0,0,0,0]])
birth_covar = np.array([[1000000,0,0,0],[0,10,0,0],[0,0,1000000,0],[0,0,0,10]])
birth_component = TaggedWeightedGaussianState(        )"""
    
"----------------------------------- Prunning --------------------------------"
from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer

# Initialise a Gaussian Mixture reducer
merge_threshold = 16
prune_threshold = 1E-6
reducer = GaussianMixtureReducer(
        prune_threshold=prune_threshold,
        pruning=True,
        merge_threshold=merge_threshold,
        merging=True)

"---------------------------- Construct the Tracker --------------------------"
"""from stonesoup.tracker.pointprocess import GMPHDTargetTracker
tracker = GMPHDTargetTracker(
        detector=detection_sim,
        hypothesiser=hypothesiser,
        updater=updater,
        reducer=reducer,
        birth_component=birth_component,
        extraction_threshold=0.5)"""

from stonesoup.tracker.pointprocess import PointProcessMultiTargetTracker

tracker = PointProcessMultiTargetTracker(
#        detector=detection_sim,
        detector = detector1,
        hypothesiser=hypothesiser,
        updater=updater,
        reducer=reducer,
        birth_component=birth_component,
        extraction_threshold=0.5)
      
"----------------------------- Running the Tracker ---------------------------"
tracks = set()
groundtruth_paths = set()  # Store for plotting later
detections = set()         # Store for plotting later

for n, (time, ctracks) in enumerate(tracker.tracks_gen(), 1):
    print("============================")    
    print("N:", n)
    print("Time:", time)
    tracks.update(ctracks)
    detections.update(tracker.detector.detections)
#    groundtruth_paths.update(tracker.detector.groundtruth.groundtruth_paths)
    if not n % 10:
        print(time, len(ctracks), len(tracks))

"""   i=1
for track in tracks:
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++ state_vectors")
    print(track.states)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print(len(track))
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++ state_vectors")
    print(track.id)
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++ state_vectors")    
    data = np.array([state.state_vector for state in track.states])
    i+=1
print(i)    


    print("# of Estimated Targets:", tracker.estimated_number_of_targets)
    """   




#"------------------------------- Plotting Results ----------------------------"
#
#kalman_tracks = {track for track in tracks}
#print(kalman_tracks)
#
#
#from matplotlib import pyplot as plt
#plt.rcParams['figure.figsize'] = (14, 12)
#plt.style.use('seaborn-colorblind')
#
#from stonesoup.metricgenerator.plotter import TwoDPlotter
#plot_data = TwoDPlotter([0, 2], [0, 2], [0, 1]).plot_tracks_truth_detections
#
#plt.rcParams['figure.figsize'] = (12, 10)
#plt.style.use('seaborn-colorblind')
#plot_data(kalman_tracks, groundtruth_paths, detections);plt.title("2D tracker");












