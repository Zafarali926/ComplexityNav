import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import HDBSCAN
from scipy.spatial import Voronoi
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from tqdm import tqdm
from itertools import combinations
import random
from numpy.linalg import norm




class Grouper():

    def __init__(self):
        self.humans = None
        self.states = None

        # for grouping
        self.group_colors = ['black']
        self.group_labels = [0]

    def _DBScan_grouping(self, labels, properties, standard):
            '''
            DBSCAN clustering using the sklearn toolbox
            Inputs:
                labels: the input labels. This will be destructively updated to
                        reflect the group memberships after DBSCAN.
                properties: the input that clustering is based on.
                            Could be positions, velocities or orientation.
                standard: the threshold value for clustering.

            Credit: Wang et al. (https://proceedings.mlr.press/v164/wang22e.html)
            '''

            max_lb = max(labels)
            for lb in range(max_lb + 1):
                sub_properties = []
                sub_idxes = []
                # Only perform DBSCAN within groups (i.e. have the same membership id)
                for i in range(len(labels)):
                    if labels[i] == lb:
                        sub_properties.append(properties[i])
                        sub_idxes.append(i)
        
                # If there's only 1 person then no need to further group
                if len(sub_idxes) > 1:
                    db = DBSCAN(eps = standard, min_samples = 1)
                    sub_labels = db.fit_predict(sub_properties)
                    max_label = max(labels)


                    # db.fit_predict always return labels starting from index 0
                    # we can add these to the current biggest id number to create
                    # new group ids.
                    for i, sub_lb in enumerate(sub_labels):
                        if sub_lb > 0:
                            labels[sub_idxes[i]] = max_label + sub_lb
            return labels


    def _HDBScan_grouping(self, labels, properties, standard):
            '''
            HDBSCAN clustering using the sklearn toolbox
            Inputs:
                labels: the input labels. This will be destructively updated to
                        reflect the group memberships after DBSCAN.
                properties: the input that clustering is based on.
                            Could be positions, velocities or orientation.
                standard: the threshold value for clustering.

            Credit: Wang et al. (https://proceedings.mlr.press/v164/wang22e.html)
            '''


            max_lb = max(labels)
            for lb in range(max_lb + 1):
                sub_properties = []
                sub_idxes = []
                # Only perform DBSCAN within groups (i.e. have the same membership id)
                for i in range(len(labels)):
                    if labels[i] == lb:
                        sub_properties.append(properties[i])
                        sub_idxes.append(i)
            
                # If there's only 1 person then no need to further group
                if len(sub_idxes) > 1:
                    hdb = HDBSCAN(min_cluster_size = 2)
                    sub_labels = hdb.fit_predict(sub_properties)
                    max_label = max(labels)


                    # db.fit_predict always return labels starting from index 0
                    # we can add these to the current biggest id number to create
                    # new group ids.
                    for i, sub_lb in enumerate(sub_labels):
                        if sub_lb > 0:
                            labels[sub_idxes[i]] = max_label + sub_lb
            return labels



    def dbscan_group(self, frame):
            # if params == None:
            pos = 1
            ori = 10
            vel = 1.0

            params = {'position_threshold': pos,
                        'orientation_threshold': ori / 180.0 * np.pi,
                        'velocity_threshold': vel,
                        'velocity_ignore_threshold': 0.5}


            num_people = len(self.humans)
            vel_orientation_array = []
            vel_magnitude_array = []
            position_array = []
            humans_at_frame = [self.states[frame][1][j] for j in range(len(self.humans))]

            # logging.info("Group at frame: {}".format(frame))
            for h in humans_at_frame:
                vx = h.vx
                vy = h.vy
                velocity_magnitude = np.sqrt(vx ** 2 + vy ** 2)
                position_array.append((h.px,h.py))
                if velocity_magnitude < params['velocity_ignore_threshold']:
                    # if too slow, then treated as being stationary
                    vel_orientation_array.append((0.0, 0.0))
                    vel_magnitude_array.append((0.0, 0.0))
                else:
                    vel_orientation_array.append((vx / velocity_magnitude, vy / velocity_magnitude))
                    vel_magnitude_array.append((0.0, velocity_magnitude)) # Add 0 to fool DBSCAN
            # grouping in current frame (three passes, each on different criteria)
            labels = [0] * num_people
            labels = self._DBScan_grouping(labels, vel_orientation_array,
                                        params['orientation_threshold'])
            labels = self._DBScan_grouping(labels, vel_magnitude_array,
                                        params['velocity_threshold'])
            labels = self._DBScan_grouping(labels, position_array,
                                    params['position_threshold'])
        
            # labels = self._HDBScan_grouping(labels, vel_orientation_array,
            #                             params['orientation_threshold'])
            # labels = self._HDBScan_grouping(labels, vel_magnitude_array,
            #                             params['velocity_threshold'])
            # labels = self._HDBScan_grouping(labels, position_array,
            #                             params['position_threshold'])
            return labels

    
    def pair2cluster(self, pairwiseData,totalNum):

        clusterNum = 0
        clusterIndexList = [0] * totalNum

        for i in range(len(pairwiseData)):
            curPair = pairwiseData[i]
            curPairAlabel = clusterIndexList[curPair[0]]
            curPairBlabel = clusterIndexList[curPair[1]]

            if curPairAlabel == 0 and curPairBlabel == 0:
                clusterNum = clusterNum + 1
                curPairLabel = clusterNum
                clusterIndexList[curPair[0]] = curPairLabel
                clusterIndexList[curPair[1]] = curPairLabel
            elif curPairAlabel != 0 and curPairBlabel == 0:
                clusterIndexList[curPair[1]] = curPairAlabel
            elif curPairBlabel != 0 and curPairAlabel == 0:
                clusterIndexList[curPair[0]] = curPairBlabel
            else:
                combineLabel = min(curPairAlabel,curPairBlabel)
                for j in range(len(clusterIndexList)):
                    if clusterIndexList[j] == curPairAlabel or clusterIndexList[j] == curPairBlabel:
                        clusterIndexList[j] = combineLabel
        return clusterIndexList



    def CF_neighbor2pair(self, neighborSet,correlationSet,d):
        zero_neighbor_set = neighborSet[1]
        n_point = zero_neighbor_set.shape[0]

        pair_set = []
        corre_set = []

        for i in range(n_point):
            cur_intersect = zero_neighbor_set[i]
            cur_corre = [0] * len(zero_neighbor_set[0])
            for j in range(1,d):
                next_neighbor_set = neighborSet[j][i]
                next_corre_set = correlationSet[j][i]
                cur_intersect, cur_indexes, next_indexes = np.intersect1d(cur_intersect, next_neighbor_set, return_indices=True)
                tempList = []
                for k in range(len(cur_indexes)):
                    tempList.append(cur_corre[cur_indexes[k]] + next_corre_set[next_indexes[k]]) 
                cur_corre = tempList
            if cur_intersect.size > 0:
                for k in range(len(cur_corre)):
                    corre_set.append(cur_corre[k])
                for k in range(len(cur_intersect)):
                    pair_set.append([i, cur_intersect[k]])

        return pair_set, corre_set


        

    def CF_neighbor(self, allXset, allVset, K, id_set = None):
        d = len(allXset)
        n_point = len(allXset[0])
        neighbor_set = [None] * d
        correlation_set = [None] * d

        angle_threshold = 45

        for j in range(d):
            cur_allX = allXset[j]
            cur_allV = allVset[j]
            cur_neighbor_graph = np.zeros((n_point, K), dtype=int)
            cur_correlation_graph = np.zeros((n_point, K))

            for i in range(n_point):
                cur_X = cur_allX[i]
                diff = np.tile(cur_X, [n_point, 1]) - cur_allX
                distance = [[x, 0] for x in range(n_point)]
                for k, (x,y) in enumerate(diff):
                    distance[k][1] = np.sqrt(x**2 + y**2)

                distance = sorted(distance, key=lambda x: x[1])  

                kNN = 0
                ind = 0
                while kNN < K and ind < len(distance):
                    if cur_allV[distance[kNN+1][0]][0] != 0 and (np.arctan(cur_allV[distance[kNN+1][0]][1]/cur_allV[distance[kNN+1][0]][0]) * np.pi/180) < angle_threshold:
                        if norm(cur_allV[i]) > 0 and norm(cur_allV[distance[kNN + 1][0]]) > 0:
                            coefficient = np.dot(cur_allV[i], cur_allV[distance[kNN + 1][0]])
                            coefficient /= (norm(cur_allV[i]) * norm(cur_allV[distance[kNN + 1][0]]))
                            cur_neighbor_graph[i][kNN] = distance[kNN + 1][0]
                            cur_correlation_graph[i][kNN] = coefficient
                            kNN += 1
                    ind += 1
                    kNN += 1
            
            neighbor_set[j] = cur_neighbor_graph
            correlation_set[j] = cur_correlation_graph

        return neighbor_set, correlation_set


    def getPosVel(self, frame, numFrames):
        
        
        firstFrame = max(0,frame - numFrames)
        
        sz = 0
        for i in range(firstFrame, frame):
            if sz < len(self.states[i][1]):
                sz = len(self.states[i][1]) 

        positions = np.empty((frame-firstFrame,sz,2),np.float64)
        velocities = np.empty((frame-firstFrame,sz,2),np.float64)

        for i in range(firstFrame, frame):
            humans_at_frame = [self.states[i][1][j] for j in range(len(self.humans))]
            for j,h in enumerate(humans_at_frame):
                positions[i - firstFrame][j][0]= h.px
                positions[i - firstFrame][j][1] = h.py

                velocities[i - firstFrame][j][0] = h.vx
                velocities[i - firstFrame][j][1] = h.vy


        return positions, velocities



    def CoherentFilter(self, frame, numFrames, K, lamda):
        '''
        Adapted from Algorithm 1 of "Coherent Filtering: Detecting coherent motion patterns at each frame"
        by Zhou et al. (https://scispace.com/pdf/coherent-filtering-detecting-coherent-motions-from-crowd-xi8nhnf2s2.pdf)
        '''

        ## step1: find K nearest neighbor set at each time
        positions, velocities = self.getPosVel(frame, numFrames) # For ORCA

        numFrames = min(frame, numFrames)
        firstPos = positions[0]
        nPoint = len(firstPos)


        neighborSet, correlationSet = self.CF_neighbor(positions,velocities,K)


        ## step2: find the invariant neighbor and pairwise connection set
        [pairSet,correSet] = self.CF_neighbor2pair(neighborSet,correlationSet,numFrames)


        ## step3: threshold pairwise connection set by the averaged correlation values, then generate cluster components 
        pairIndex = [] # included pairwise connection
        for i in range(len(correSet)):
            if correSet[i] > lamda:
                pairIndex.append(i)
        
        filteredPairs = [pairSet[i] for i in pairIndex]
        # 

        
        clusterIndex = self.pair2cluster(filteredPairs,nPoint)


        return clusterIndex



    
    def group_color(self, frame):
        '''
        Generates social groups of pedestrians using either DBScan or CoherentFilter (see comment below). 
        
        '''

        if frame <= 3:
            return ['b'] * 100
        
        ###############################################################
        ## switch between coherent filter and dbscan. 
        ## Note: only coherent filter requires the above frame buffer.

        # self.group_labels = self.dbscan_group(frame) 
        self.group_labels = self.CoherentFilter(frame, 5, 3, 0.9)

        ###############################################################


        def get_colors(n):
            colors = []
            for i in range(n):
                new_c = "#%06x" % random.randint(0, 0xFFFFFF)
                for c in self.group_colors:
                    while (np.abs(c - new_c) < 0xFFFFF0):
                        new_c = "#%06x" % random.randint(0, 0xFFFFFF)
                colors.append(new_c)
            return colors
        
        
        get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
        num_groups = max(self.group_labels) + 1

        if num_groups > len(self.group_colors):
            self.group_colors = self.group_colors + get_colors(num_groups - len(self.group_colors))

        group_membership_color = [0] * len(self.group_labels)

        for i in range(len(self.group_labels)):
           group_membership_color[i] = self.group_colors[self.group_labels[i]]
        return group_membership_color


class GroupSpaceGenerator():
    def __init__(self):
        pass

    def boundary_dist(self, velocity, rel_ang, const=0.354163):
        # Parameters from Rachel Kirby's thesis
        front_coeff = 1.0
        side_coeff = 2.0 / 3.0
        rear_coeff = 0.5
        safety_dist = 0.5
        velocity_x = velocity[0]
        velocity_y = velocity[1]

        velocity_magnitude = np.sqrt(velocity_x ** 2 + velocity_y ** 2)
        variance_front = max(0.5, front_coeff * velocity_magnitude)
        variance_side = side_coeff * variance_front
        variance_rear = rear_coeff * variance_front

        rel_ang = rel_ang % (2 * np.pi)
        flag = int(np.floor(rel_ang / (np.pi / 2)))
        if flag == 0:
            prev_variance = variance_front
            next_variance = variance_side
        elif flag == 1:
            prev_variance = variance_rear
            next_variance = variance_side
        elif flag == 2:
            prev_variance = variance_rear
            next_variance = variance_side
        else:
            prev_variance = variance_front
            next_variance = variance_side

        dist = np.sqrt(const / ((np.cos(rel_ang) ** 2 / (2 * prev_variance)) + (np.sin(rel_ang) ** 2 / (2 * next_variance))))
        dist = max(safety_dist, dist)


        return dist

    def draw_social_shapes(self, position, velocity, const=0.35):
        '''
        draws social group shapes given the positions and velocities of the pedestrians, 
        with social considerations based on the work by Rachel Kirby (https://www.ri.cmu.edu/publications/social-robot-navigation/)

        returns the vertices of the convex hull
        '''

        total_increments = 10 # controls the resolution of the blobs
        quater_increments = total_increments / 4
        angle_increment = 2 * np.pi / total_increments

        # Draw a personal space for each pedestrian within the group
        contour_points = []
        for i in range(len(position)):
            center_x = position[i][0]
            center_y = position[i][1]
            velocity_x = velocity[i][0]
            velocity_y = velocity[i][1]
            velocity_angle = np.arctan2(velocity_y, velocity_x)

            # Draw four quater-ovals with the axis determined by front, side and rear "variances"
            # The overall shape contour does not have discontinuities.
            for j in range(total_increments):

                rel_ang = angle_increment * j
                value = self.boundary_dist(velocity[i], rel_ang, const)
                addition_angle = velocity_angle + rel_ang
                x = center_x + np.cos(addition_angle) * value
                y = center_y + np.sin(addition_angle) * value
                contour_points.append([x, y])

        # Get the convex hull of all the personal spaces
        convex_hull_vertices = []
        hull = ConvexHull(np.array(contour_points))
        for i in hull.vertices:
            hull_vertice = [contour_points[i][0], contour_points[i][1]]
            convex_hull_vertices.append(hull_vertice)

        return convex_hull_vertices



    def trace_polygon(self, groups, group_positions, g_index):
        '''
        Finds and returns the perimeter of the specified group (g_index). 
        Excess information passed as parameters for simplicity, could certainly be optimized.
        '''

        threshold = 1

        def find_right_side(g_ind):
            side = [] # bottom to top
            g_indices_w_y = [] # indices of individuals in the GROUPS OBJECT, so not the ind of the agent, but the ind of the ind of the agent in the group
            for i in range(len(groups[g_ind])):
                g_indices_w_y.append([i, group_positions[g_ind][i][1]])

            sorted_indices = sorted(g_indices_w_y, key=lambda x: x[1])
            # logging.info('sorted: {}'.format(sorted_indices))

            for pair in sorted_indices:
                onRight = True
                i = pair[0]
                for j in range(len(groups[g_ind])):
                    if j != i and group_positions[g_ind][j][0] > group_positions[g_ind][i][0] and np.abs(group_positions[g_ind][j][1] - group_positions[g_ind][i][1]) < threshold:
                        onRight = False
                if onRight:
                    side.append(i)
            

            return side
        
        def find_top(g_ind):
            side = [] # bottom to top
            g_indices_w_x = [] # indices of individuals in the GROUPS OBJECT, so not the ind of the agent, but the ind of the ind of the agent in the group
            for i in range(len(groups[g_ind])):
                g_indices_w_x.append([i, group_positions[g_ind][i][0]])

            sorted_indices = sorted(g_indices_w_x, key=lambda x: x[1])

            for pair in reversed(sorted_indices):
                onRight = True
                ind = pair[0]
                for j in range(len(groups[g_ind])):
                    if j != ind and group_positions[g_ind][j][1] > group_positions[g_ind][ind][1] and np.abs(group_positions[g_ind][j][0] - group_positions[g_ind][ind][0]) < threshold:
                        onRight = False
                if onRight:
                    side.append(ind)
            
            return side

        def find_left_side(g_ind):
            side = [] # bottom to top
            g_indices_w_y = [] # indices of individuals in the GROUPS OBJECT, so not the ind of the agent, but the ind of the ind of the agent in the group
            for i in range(len(groups[g_ind])):
                g_indices_w_y.append([i, group_positions[g_ind][i][1]])

            sorted_indices = sorted(g_indices_w_y, key=lambda x: x[1])

            for pair in reversed(sorted_indices):
                onRight = True
                ind = pair[0]
                for j in range(len(groups[g_ind])):
                    if j != ind and group_positions[g_ind][j][0] < group_positions[g_ind][ind][0] and np.abs(group_positions[g_ind][j][1] - group_positions[g_ind][ind][1]) < threshold:
                        onRight = False
                if onRight:
                    side.append(ind)
            
            return side
        
        def find_bottom(g_ind):
            side = [] # bottom to top
            g_indices_w_x = [] # indices of individuals in the GROUPS OBJECT, so not the ind of the agent, but the ind of the ind of the agent in the group
            for i in range(len(groups[g_ind])):
                g_indices_w_x.append([i, group_positions[g_ind][i][1]])

            sorted_indices = sorted(g_indices_w_x, key=lambda x: x[1])

            for pair in sorted_indices:
                onRight = True
                i = pair[0]
                for j in range(len(groups[g_ind])):
                    if j != i and group_positions[g_ind][j][1] < group_positions[g_ind][i][1] and np.abs(group_positions[g_ind][j][0] - group_positions[g_ind][i][0]) < threshold:
                        onRight = False
                if onRight:
                    side.append(i)
            
            return side
        
        right = find_right_side(g_index)
        top = find_top(g_index) 
        left = find_left_side(g_index) 
        bottom = find_bottom(g_index)

        return right, top, left, bottom

