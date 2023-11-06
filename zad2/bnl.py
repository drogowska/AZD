import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import random
import seaborn as sns
from sklearn.manifold import TSNE


def distance_euclidean(center, point_obj):
    """ Calculate the Euclidean Distance between 2 pair of points"""
    y_dist = math.pow(point_obj[1] - center[1], 2)
    x_dist = math.pow(point_obj[0] - center[0], 2)
    distance = math.sqrt(y_dist + x_dist)
    return distance


def sort(input_list):
    """Function to Remove Duplicate Elements in list"""
    sorted_list = []
    for i in input_list:
        if i not in sorted_list:
            sorted_list.append(i)
    return sorted_list


def nested_loop(first_array_block, list_of_other_blocks, count_threshold, radius_input):
    """Function to carry out the nested loop algorithm for every Block"""
    list_possible_outliers = []
    ### This section is to determine the possible Outliers from the First Array Block
    # - Checking each points in the First Array against every other point in the First Array
    for coor1 in first_array_block:
        count_coor1 = 0  # this is to count the number of neighbouring points for the current Center Point (coor1)
        count_iteration = 0  # to track the number of points that were checked
        for coor2 in first_array_block:
            count_iteration += 1
            ### Determine the distance between 2 points (coor1, coor2) 
            distance_coor1_coor2 = distance_euclidean(coor1, coor2)
            ### If the distance between 2 points is within the Distance Threshold
            # - consider this point to be a neighbouring point
            if distance_coor1_coor2 <= radius_input:
                count_coor1 += 1  # Increment 1 to number of neighbouring points variable
            ### When the number of neighbouring points exceed the threshold (for minimum number of nearby points)
            # - the Point (coor1) is considered not an Outlier and the for loop is ended
            # - Proceed to the next Point (coor1)
            if count_coor1 > count_threshold:
                break
            ### If every point (coor2) has been checked against the current point (coor1)
            # - if the number of neighbouring points is less than the threshold
            # - consider this point (coor1) a possible outlier 
            if count_coor1 <= count_threshold and count_iteration == len(first_array_block):
                list_possible_outliers.append(coor1)  # store (coor1) Point into the possible outlier list

    ### This section is to determine the possible Outliers (found from previous section) against the other Blocks 
    # - if a point is still considered to be an Outlier 
    # - the coordinates of the Outlier Points are Returned by the Function
    # Variable to output the final list of Points of Outliers Found
    list_possible_final_outliers = []
    for coor1_next in list_possible_outliers:
        ### This section is to check, for every possible Outlier Points from the Previous Section
        # - Compare the distance of the Outlier Points to the other Points in the Other Blocks (Blocks besides the First Array Block)
        # - Determine if the Outlier Point is still an Outlier
        for block in list_of_other_blocks:
            count_coor1_next = 0  # this is to count the number of neighbouring points for the current Center Point (coor1_next)
            count_iteration = 0  # to track the number of points that were checked
            for coor2_next in block:
                count_iteration += 1
                ### Determine the distance between 2 points (coor1_next, coor2_next)
                distance_coor1_coor2_next = distance_euclidean(coor1_next, coor2_next)
                ### If the distance between 2 points is within the Distance Threshold
                # - consider this point to be a neighbouring point
                if distance_coor1_coor2_next <= radius_input:
                    count_coor1_next += 1  # Increment 1 to number of neighbouring points variable
                ### When the number of neighbouring points exceed the threshold (for minimum number of nearby points)
                # - The current point is not an outlier (coor1_next) 
                # - Proceed to the next Point (coor1_next)
                if count_coor1_next > count_threshold:
                    break
                ### If every point (coor2_next) has been checked against the current point (coor1_next)
                # - if the number of neighbouring points is less than the threshold
                # - consider this point (coor1_next) an Outlier
                if count_coor1_next <= count_threshold and count_iteration == len(block):
                    list_possible_final_outliers.append(coor1_next)
        list_possible_final_outliers = sort(list_possible_final_outliers)  # Remove any duplicate elements

    return list_possible_final_outliers

def BNL(X_test):
    tsne = TSNE(n_components=2, random_state=42, perplexity=20)
    tsne_results = tsne.fit_transform(X_test)

    split_blocks = np.array_split(tsne_results, 4)
    coordinate_A = [list(x) for x in split_blocks[0]]
    coordinate_B = [list(x) for x in split_blocks[1]]
    coordinate_C = [list(x) for x in split_blocks[2]]
    coordinate_D = [list(x) for x in split_blocks[3]]

    ### Set the Parameters and Threshold
    stages = 4  # Fixed
    num_points_threshold = 3  # <==
    distance_threshold = 3  # <==
    all_outlier_list = []  # Records all the Outlier Points determined from each Stage

    ### Start the Nested Loop Algorithm Outlier Detection
    for stage in range(1, stages + 1):
        if stage == 1:
            ### Block A
            all_outlier_list += nested_loop(coordinate_A, [coordinate_B, coordinate_C, coordinate_D],
                                            num_points_threshold,
                                            distance_threshold)

        if stage == 2:
            ### Block D
            all_outlier_list += nested_loop(coordinate_D, [coordinate_B, coordinate_C], num_points_threshold,
                                            distance_threshold)

        if stage == 3:
            ### Block C
            all_outlier_list += nested_loop(coordinate_C, [coordinate_A, coordinate_B], num_points_threshold,
                                            distance_threshold)

        if stage == 4:
            ### Block B
            all_outlier_list += nested_loop(coordinate_B, [coordinate_D, coordinate_A], num_points_threshold,
                                            distance_threshold)

    dataset_y = [x[1] for x in tsne_results]
    dataset_x = [x[0] for x in tsne_results]
    dataset_yo = [x[1] for x in all_outlier_list]
    dataset_xo = [x[0] for x in all_outlier_list]
    tsne_dfo = pd.DataFrame({"Dimensionoa": dataset_xo, "Dimensionob": dataset_yo})
    tsne_df = pd.DataFrame(tsne_results, columns=['Dimensiona', 'Dimensionb'])
    s = tsne_df.merge(tsne_dfo, left_on=['Dimensiona', 'Dimensionb'], right_on=['Dimensionoa', 'Dimensionob'],
                      how='left')
    s["label"] = s.Dimensionoa.notnull().astype(int)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='Dimensiona', y='Dimensionb', hue='label', data=s, palette='viridis')
    plt.show()





