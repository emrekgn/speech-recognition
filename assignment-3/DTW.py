# -*- coding: utf-8 -*-

import numpy


class DTW(object):
    """
    a class to do DTW(dynamic time warping) and Viterbi Search between mfcc features
    """

    def __init__(self, templates):
        """
        :param templates: a list include template features
        """
        self.threshold = 0
        self.templates_length_list = []
        self.templates = templates[:]
        self.begin_distance = []
        self.temp_templates_added_length_list = []
        self.template = []
        for i in xrange(0, len(self.templates)):
            self.templates[i].insert(0, list(numpy.zeros(len(templates[i][0]))))
            self.template.extend(self.templates[i])
            self.templates_length_list.append(len(self.templates[i]))
            if i == 0:
                self.temp_templates_added_length_list.append(len(self.templates[i]))
            else:
                self.temp_templates_added_length_list.append(
                    len(self.templates[i]) + self.temp_templates_added_length_list[i - 1])
        self.temp_templates_added_length_list.append(0)
        self.distance_matrix_rows = len(self.template)
        self.templates_added_length_list = [0 for i in xrange(0, self.distance_matrix_rows + 1)]
        for i in self.temp_templates_added_length_list:
            self.templates_added_length_list[i] = 1
        self.temp_distance = [float('inf') for i in xrange(0, self.distance_matrix_rows)]
        for i in xrange(0, len(self.templates_length_list)):
            self.begin_distance.extend([0])
            self.begin_distance.extend([float('inf') for i in xrange(1, self.templates_length_list[i])])
        self.old_extensible_index = [1 for i in xrange(0, self.distance_matrix_rows)]


    def DTW(self, input_feature, strategy, accuracy=1.0, cost_function=0, covariance_matrix=[], edge_cost=[]):
        """
        a routine to compute the DTW between mfcc features
        :param input_feature: list,the input mfcc feature
        :param strategy: 0:don't pruning
                         1:use relative pruning
        :param accuracy: self.threshold = 2 * min(new_distance) / max(0.00001, accuracy)
        :param cost_function: 0:use euclidean_distance as the distance
                              1:use DTW as Viterbi Search,which has a edge cost and a node cost
        :param: covariance_matrix, a list,covariance_matrix[i] represent for ith state's covariance matrix
        :param: edge_cost[i][j] represent for the cost transform from state i to j
        :return: min_distance: minimum distance between input feature and multiple templates
        :return: template: index pf the template that has minimum distance
        :return: path: path from begin to the end
        """
        input_feature.insert(0,
                             list(
                                 numpy.zeros(len(input_feature[0]))))  # to avoid the bug when compare 'this' with 'his'
        column = self.template  # template length equals to column length
        row = input_feature
        distance_matrix_columns = len(row)
        distance = self.begin_distance
        old_extensible_index = self.old_extensible_index[:]
        last_block_position = [[0 for i in xrange(self.distance_matrix_rows)] for j in
                               xrange(distance_matrix_columns)]
        for i in xrange(1, distance_matrix_columns):
            new_extensible_index = [0 for j in xrange(0, self.distance_matrix_rows)]
            new_distance = self.temp_distance[:]
            for j in xrange(0, self.distance_matrix_rows):
                if old_extensible_index[j] or not strategy:
                    if cost_function == 0:
                        self.get_distance_using_euclidean_distance_as_node_cost(new_distance, last_block_position,
                                                                                row, column, distance, i, j)
                    else:
                        self.get_distance_doing_viterbi_search(new_distance, last_block_position,
                                                               row, column, distance, i, j, covariance_matrix,
                                                               edge_cost)
            if strategy == 1:
                self.threshold = 2 * min(new_distance) / max(0.00001, accuracy)
                for j in xrange(0, self.distance_matrix_rows):
                    if new_distance[j] <= self.threshold:
                        new_extensible_index[j] = 1
                        if not self.templates_added_length_list[j + 1]:
                            new_extensible_index[j + 1] = 1
                        if j + 2 < self.distance_matrix_rows and not self.templates_added_length_list[j + 2]:
                            new_extensible_index[j + 2] = 1
                old_extensible_index = new_extensible_index[:]
            # print distance
            distance = new_distance[:]
        # print distance
        total_distance = [distance[i - 1] for i in self.temp_templates_added_length_list[:-1]]
        min_distance = min(total_distance)
        j = self.temp_templates_added_length_list[total_distance.index(min_distance)] - 1
        i = distance_matrix_columns - 1
        path = []
        while last_block_position[i][j] != 0:
            last_position = last_block_position[i][j][:]
            path.append(last_position)
            i = last_position[0]
            j = last_position[1]
        return min_distance, total_distance.index(min_distance), path


    def get_distance_using_euclidean_distance_as_node_cost(self, new_distance, last_block_position, row, column,
                                                           distance, i, j):
        """
        update new_distance list and last_block_position list using euclidean distance as node cost
        """
        cost = self.euclidean_distance(row[i], column[j])
        if self.templates_added_length_list[j]:
            new_distance[j] = distance[j] + cost
            last_block_position[i][j] = [i - 1, j]
        elif self.templates_added_length_list[j - 1]:
            if distance[j] > distance[j - 1]:
                new_distance[j] = distance[j - 1] + cost
                last_block_position[i][j] = [i - 1, j - 1]
            else:
                new_distance[j] = distance[j] + cost
                last_block_position[i][j] = [i - 1, j]
        else:
            new_distance[j] = min(distance[j], distance[j - 1], distance[j - 2])
            index = [distance[j], distance[j - 1], distance[j - 2]].index(
                new_distance[j])
            new_distance[j] += cost
            if index == 0:
                last_block_position[i][j] = [i - 1, j]
            elif index == 1:
                last_block_position[i][j] = [i - 1, j - 1]
            else:
                last_block_position[i][j] = [i - 1, j - 2]


    def get_distance_doing_viterbi_search(self, new_distance, last_block_position, frames, mean,
                                          distance, i, j, covariance_matrix, edge_cost):
        """
        update new_distance list and last_block_position list doing viterbi search which has a node cost and a edge cost
        :param: covariance_matrix, a array,covariance_matrix[i] represent for ith state's covariance matrix
        :param: mean,a list,mean[i] represent for ith state's mean
        :param: i,j,compare ith frame with jth state
        :param edge_cost[i][j] represent for the cost transform from state i to j
        """
        if j == 0:
            node_cost = 0
        #print 'node_cost', i, j, node_cost
        if self.templates_added_length_list[j]:
            cost_from_left_node = node_cost + edge_cost[j][j]
            new_distance[j] = distance[j] + cost_from_left_node
            last_block_position[i][j] = [i - 1, j]
        elif self.templates_added_length_list[j - 1]:
            cost_from_left_node = node_cost + edge_cost[j][j]
            cost_from_left_down_node = node_cost + edge_cost[j - 1][j]
            if distance[j] + cost_from_left_node > distance[j - 1] + cost_from_left_down_node:
                new_distance[j] = distance[j - 1] + cost_from_left_down_node
                last_block_position[i][j] = [i - 1, j - 1]
            else:
                new_distance[j] = distance[j] + cost_from_left_node
                last_block_position[i][j] = [i - 1, j]
        else:
            cost_from_left_node = node_cost + edge_cost[j][j]
            cost_from_left_down_node = node_cost + edge_cost[j - 1][j]
            cost_from_left_down_down_node = node_cost + edge_cost[j - 2][j]
            new_distance[j] = min(distance[j] + cost_from_left_node, distance[j - 1] + cost_from_left_down_node, \
                                  distance[j - 2] + cost_from_left_down_down_node)
            index = [distance[j] + cost_from_left_node, distance[j - 1] + cost_from_left_down_node, \
                     distance[j - 2] + cost_from_left_down_down_node].index(new_distance[j])
            if index == 0:
                last_block_position[i][j] = [i - 1, j]
            elif index == 1:
                last_block_position[i][j] = [i - 1, j - 1]
            else:
                last_block_position[i][j] = [i - 1, j - 2]


    def euclidean_distance(self, feature1, feature2):
        """
        compute euclidean distance between two vectors
        :param feature1:ndarray,vector 1
        :param feature2:ndarray,vector 2
        :return:distance between them
        """
        return numpy.linalg.norm(numpy.subtract(feature1, feature2))
