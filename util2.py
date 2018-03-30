"""Collection of utility functions."""
#Data: http://www2.informatik.uni-freiburg.de/~stachnis/datasets.html


import math
import numpy as np
import random
import time

from sklearn.metrics import roc_auc_score, roc_curve
import scipy as sp

#import hilbertMapSparseMethod3_res as hm


class Timing(object):

    """Allows timing the runtime of code segments."""

    def __init__(self):
        """Creates a new Timing instance."""
        self._start = time.time()
        self._end = None
        self._count = 0

    def update(self, count=1):
        """Updates the end time and increments the count.

        :param count the number of operations performed
        """
        self._count += count
        self._end = time.time()

    def diff(self):
        """Returns the duration captured by the Timing instance.

        :return time elapsed between start and end
        """
        return self._end - self._start

    def average(self):
        """Returns the average duration of an operation.

        :return average duration of a single operation
        """
        return self.diff() / self._count

    def reset(self):
        """Resets the Timing instance to its initial state."""
        self._start = time.time()
        self._end = None
        self._count = 0


def normalize_angle(angle):
    """Normalizes the angle to the range [-PI, PI].

    :param angle the angle to normalize
    :return normalized angle
    """
    center = 0.0
    n_angle = angle - 2 * math.pi * math.floor((angle + math.pi - center) / (2 * math.pi))
    assert(-math.pi <= n_angle <= math.pi)

    return n_angle


def bresenham(start_point, end_point):
    """Returns the points on the line from start to end point.

    :params start_point the start point coordinates
    :params end_point the end point coordinates
    :returns list of coordinates on the line from start to end
    """
    coords = []

    dx = abs(end_point[0] - start_point[0])
    dy = abs(end_point[1] - start_point[1])
    x, y = start_point[0], start_point[1]
    sx = -1 if start_point[0] > end_point[0] else 1
    sy = -1 if start_point[1] > end_point[1] else 1
    if dx > dy:
        err = dx / 2.0
        while x != end_point[0]:
            coords.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != end_point[1]:
            coords.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy        
    coords.append((x, y))

    return coords


def bounding_box(data, padding=5.0):
    """Returns the bounding box to the given 2d data.

    :param data the data for which to find the bounding box
    :param padding the amount of padding to add to the extreme values
    :return x and y limits
    """
    dimensions = len(data[0])
    limits = []
    for dim in range(dimensions):
        limits.append((
            np.min([entry[dim] for entry in data]) - padding,
            np.max([entry[dim] for entry in data]) + padding
        ))
    assert(len(limits) > 1)
    print('limits', limits)
    return limits[0], limits[1]


def perturb_data(poses):
    """Returns perturbed position information.

    Adds a small amount of noise to position and orientation.

    :param poses the list of poses to perturb
    :return pose information with additional noise
    """
    new_poses = []
    for pose in poses:
        dx = random.gauss(0.0, 0.1)
        dy = random.gauss(0.0, 0.1)
        dtheta = random.gauss(0.0, 0.02)

        new_poses.append((
            pose[0] + dx,
            pose[1] + dy,
            pose[2] + dtheta
        ))
    return new_poses


def parse_carmen_log(fname):
    """Parses a CARMEN log file and extracts poses and laser scans.

    :param fname the path to the log file to parse
    :return poses and scans extracted from the log file
    """
    poses = []
    scans = []
    for line in open(fname):
        if line.startswith("FLASER"):
            arr = line.split()

            count = len(arr)-2#int(arr[1]) #180, 1081 TODO: not to hard code
            poses.append([float(v) for v in arr[-9:-6]])
            scans.append([float(v) for v in arr[2:2+count]])

    return poses, scans


def free_space_points(distance, pose, angle):
    """Samples points randomly along a scan ray.

    :param distance length of the ray
    :param pose the origin of the ray
    :param angle the angle of the ray from the position
    :return list of coordinates in free space based on the data
    """
    points = []
    count = max(1, int(distance / 2))
    for _ in range(count):
        r = random.uniform(0.0, max(0.0, distance-0.1))
        #r = np.clip(distance - np.random.rayleigh(10.0, 1), 0, max(0.0, distance-0.1))
        points.append([
                pose[0] + r * math.cos(angle),
                pose[1] + r * math.sin(angle)
        ])
    return points


def sampling_coordinates(x_limits, y_limits, count):
    """Returns an array of 2d grid sampling locations.

    :params x_limits x coordinate limits
    :params y_limits y coordinate limits
    :params count number of samples along each axis
    :return list of sampling coordinates
    """
    coords = []
    for i in np.linspace(x_limits[0], x_limits[1], count):
        for j in np.linspace(y_limits[0], y_limits[1], count):
            coords.append([i, j])
    return np.array(coords)

def sampling_coordinates_rand(x_limits, y_limits, count):
    """Returns an array of 2d grid sampling locations.

    :params x_limits x coordinate limits
    :params y_limits y coordinate limits
    :params count number of samples along each axis
    :return list of sampling coordinates
    """
    coords = []
    for i in np.linspace(x_limits[0], x_limits[1], count):
        for j in np.linspace(y_limits[0], y_limits[1], count):
            x = np.random.random()*80*2 - 40*2
            y = np.random.random()*50*2 - 10*2
            coords.append([x, y])
    return np.array(coords)


def create_test_train_split(logfile, percentage=0.1, sequence_length=40):
    """Creates a testing and training dataset from the given logfile.

    :param logfile the file to parse
    :param percentage the percentage to use for testing
    :param sequence_length the number of subsequent scans to remove for
        the testing data
    :return training and testing datasets containing the posts and scans
    """
    # Parse the logfile
    poses, scans = parse_carmen_log(logfile)
    
    # Create training and testing splits
    groups = int((len(poses)*percentage) / sequence_length)
    test_indices = []
    group_count = 0
    while group_count < groups:
        start = random.randint(0, len(poses)-sequence_length)
        if start in test_indices or (start+sequence_length) in test_indices:
            continue

        test_indices.extend(range(start, start+sequence_length))
        group_count += 1

    training = {"poses": [], "scans": []}
    testing = {"poses": [], "scans": []}

    for i in range(len(poses)):
        if i in test_indices:
            testing["poses"].append(poses[i])
            testing["scans"].append(scans[i])
        else:
            training["poses"].append(poses[i])
            training["scans"].append(scans[i])

    return training, testing


def sparsify_scans(logfile, percent_removed):
    """Removes a fixed percentage of readings from every scan.

    :param logfile the file to parse and sparsify
    :param percent_removed the percentage of readings to remove per scan
    :return lists of poses, training readings and test readings
    """
    assert(0 <= percent_removed <= 1)

    poses, scans = parse_carmen_log(logfile)
    discard_count = int(len(scans[0]) * percent_removed)
    angle_increment = math.pi / len(scans[0])

    train_scans = []
    test_scans = []

    for pose, ranges in zip(poses, scans):
        discard_indices = random.sample(range(len(ranges)), discard_count)
        train_ranges = []
        test_ranges = []

        for i, dist in enumerate(ranges):
            angle = normalize_angle(
                    pose[2] - math.pi + i * angle_increment + (math.pi / 2.0)
            )
            if i not in discard_indices:
                train_ranges.append((dist, angle))
            else:
                test_ranges.append((dist, angle))

        train_scans.append(train_ranges)
        test_scans.append(test_ranges)
    return poses, train_scans, test_scans


def sparsify_data(scan_data, percent_removed):
    """Removes a specified percentage of data from a dataset.

    :param scan_data the dataset to sparsify
    :param percent_removed the percentage of the data to remove
    :return the dataset where a specified percentage has been removed
    """
    assert(0 <= percent_removed <= 1)

    discard_count = int(len(scan_data[0]) * percent_removed)
    new_data = []
    for data in scan_data:
        discard_indices = random.sample(range(len(data)), discard_count)
        new_pairs = []
        for i, entry in enumerate(data):
            if i not in discard_indices:
                new_pairs.append(entry)
        new_data.append(new_pairs)
    return new_data


def roc_evaluation(model, data):
    """Performs ROC evaluation of the hilbert map on the given data.

    :param model the hilbert map instance to evaluate
    :param data the testing data
    :return true positive rate and false positive rate for varying thresholds
    """
    test_data = []
    test_labels = []
    for t_data, t_labels in data_generator(data["poses"], data["scans"]):
        test_data.extend(t_data)
        test_labels.extend(t_labels)

    offset = 0
    predictions = []
    while offset < len(test_data):
        if isinstance(model, hm.IncrementalHilbertMap):
            query = model.sampler.transform(test_data[offset:offset+100])
            predictions.extend(model.classifier.predict_proba(query)[:, 1])
        elif isinstance(model, hm.SparseHilbertMap):
            predictions.extend(model.classify(test_data[offset:offset+100])[:, 1])
        offset += 100

    fpr, tpr, _ = roc_curve(test_labels, predictions)
    auc = roc_auc_score(test_labels, predictions)
    return tpr, fpr, auc


def roc_occupancy_grid_map(grid_map, data):
    """Performs ROC evaluation of the occupancy grid map model on the given data.

    :param grid_map the occupancy grid map to evaluate
    :param data the testing data
    :return true positive rate and false positive rate for varying thresholds
    """
    test_data = []
    test_labels = []
    for t_data, t_labels in data_generator(data["poses"], data["scans"]):
        test_data.extend(t_data)
        test_labels.extend(t_labels)

    prediction = []
    for point in test_data:
        index = grid_map.to_grid(point)
        hit = grid_map.hit[index[0], index[1]]
        free = grid_map.free[index[0], index[1]]
        if (free + hit) > 0:
            prediction.append(hit / float(hit+free))
        else:
            prediction.append(0.5)

    fpr, tpr, _ = roc_curve(test_labels, prediction)
    auc = roc_auc_score(test_labels, prediction)
    return tpr, fpr, auc


def data_generator(poses, scans, step=1):
    """Generator which returns data for each scan.

    :params poses the sequence of poses
    :params scans the sequence of scans observed at each pose
    :params step the step size to use in the iteration
    :return 2d coordinates and labels for the data generated for individual
        pose and scan pairs
    """
    angle_increment = math.pi / (len(scans[0])-1)
    print('yield: #of laser sacns=%f, angle_increment=%f deg'%(len(scans[0]), angle_increment*180/np.pi))

    for i in range(0, len(poses), step):
        pose = poses[i]
        ranges = scans[i]

        points = []
        labels = []

        for i, dist in enumerate(ranges):
            # Ignore max range readings

            angle = normalize_angle(
                    0*pose[2] + i * angle_increment #- (math.pi / 2.0)
            )

            if dist == 100:
                """
                # Add laser endpoint
                points.append([
                    0*pose[0]  + 30*math.cos(angle),
                    0*pose[1] + 30*math.sin(angle)
                ])
                labels.append(1)
                """

                #for dgm
                ranges[i] = 35
                #

                # Add in between points
                free_points = free_space_points(30, pose, angle)
                points.extend(free_points)
                for coord in free_points:
                    labels.append(0)
            else:

                if dist > 40: #changed 40 to 32
                    continue

                # Add laser endpoint
                points.append([
                    0*pose[0]  + dist*math.cos(angle),
                    0*pose[1] + dist*math.sin(angle)
                ])
                labels.append(1)

                # Add in between points
                free_points = free_space_points(dist, pose, angle)
                points.extend(free_points)
                for coord in free_points:
                    labels.append(0)

        yield np.array(points), np.array(labels), ranges

def data_generator_return(poses, scans, step=1):
    """Generator which returns data for each scan.

    :params poses the sequence of poses
    :params scans the sequence of scans observed at each pose
    :params step the step size to use in the iteration
    :return 2d coordinates and labels for the data generated for individual
        pose and scan pairs
    """
    angle_increment = math.pi / (len(scans[0])-1)
    print('ret', angle_increment)

    for i in range(0, len(poses), step):
        pose = poses[i]
        ranges = scans[i]

        points = []
        labels = []

        for i, dist in enumerate(ranges):
            angle = normalize_angle(
                0*pose[2] + i * angle_increment #- (math.pi / 2.0)
            )

            if dist == 100:
                """
                points.append([
                    0*pose[0]  + 30*math.cos(angle),
                    0*pose[1] + 30*math.sin(angle)
                ])
                labels.append(1)
                """

                # Add in between points
                free_points = free_space_points(30, pose, angle)
                points.extend(free_points)
                for coord in free_points:
                    labels.append(0)
            else:
                # Ignore max range readings
                if dist > 40: #changed 40 to 32
                    continue

                # Add laser endpoint
                points.append([
                    0*pose[0]  + dist*math.cos(angle),
                    0*pose[1] + dist*math.sin(angle)
                ])
                labels.append(1)

                # Add in between points
                free_points = free_space_points(dist, pose, angle)
                points.extend(free_points)
                for coord in free_points:
                    labels.append(0)

        return np.array(points), np.array(labels)

def data_generator_with_angles(angle, dist):

    points = []
    labels = []
    for i in range(180):
        if dist[i] == 100:
            """
            #Add laset end points
            points.append([
                30*math.cos(angle[i]),
                30*math.sin(angle[i])
            ])
            labels.append(1)
            """

            #Add in between points
            free_points = free_space_points(30, [0,0,0], angle[i]) #TODO: fake poses prepared to send to util
            points.extend(free_points)
            for coord in free_points:
                labels.append(0)
        else:
            if dist[i] > 40:
                continue

            whr = np.where(angle == angle[i])

            if min(dist[whr]) < dist[i]:
                print(whr, min(dist[whr]) < dist[i])
                continue

            #Add laset end points
            points.append([
                dist[i]*math.cos(angle[i]),
                dist[i]*math.sin(angle[i])
            ])
            labels.append(1)

            #Add in between points
            free_points = free_space_points(dist[i], [0,0,0], angle[i]) #TODO: fake poses prepared to send to util
            points.extend(free_points)
            for coord in free_points:
                labels.append(0)

    return np.array(points), np.array(labels)

def read_raw_data(poses, scans, i, step=1, laserOnly=True):
    """Generator which returns data for each scan.

    :params poses the sequence of poses
    :params scans the sequence of scans observed at each pose
    :params step the step size to use in the iteration
    :return 2d coordinates and labels for the data generated for individual
        pose and scan pairs
    """

    print('reading %dth raw data... (for comparison)'%i)
    angle_increment = math.pi / (len(scans[0])-1)

    pose = poses[i]
    ranges = scans[i]

    points = []
    labels = []

    for i, dist in enumerate(ranges):
        # Ignore max range readings
        angle = normalize_angle(
                0*pose[2] + i * angle_increment #- (math.pi / 2.0)
        )

        if dist == 100:
            if laserOnly is False:
                # Add in between points
                free_points = free_space_points(30, pose, angle)
                points.extend(free_points)
                for coord in free_points:
                    labels.append(0)
        else:
            if dist > 40: #changed 40 to 32
                continue

            # Add laser endpoint
            points.append([
                0*pose[0]  + dist*math.cos(angle),
                0*pose[1] + dist*math.sin(angle)
            ])
            labels.append(1)

            if laserOnly is False:
                #Add in between points
                free_points = free_space_points(dist, pose, angle) #TODO: fake poses prepared to send to util
                points.extend(free_points)
                for coord in free_points:
                    labels.append(0)

    return np.array(points), np.array(labels)


def log_loss(act, pred, normalize=True):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    if normalize is True:
        ll = ll * -1.0/len(act)
    return ll