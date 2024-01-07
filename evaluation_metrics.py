import numpy as np
import math 

def hhmmss_to_seconds(time):  #converts hh:mm:ss to seconds
    time = time.split(':')
    return int(time[0])*3600 + int(time[1])*60 + int(time[2])

def get_temporal_iou(actual_interval, predicted_interval):
    #actual_interval = [start, end]
    #predicted_interval = [start, end]
    actual_interval = [hhmmss_to_seconds(actual_interval[0]), hhmmss_to_seconds(actual_interval[1])]
    predicted_interval = [hhmmss_to_seconds(predicted_interval[0]), hhmmss_to_seconds(predicted_interval[1])]
    intersection = max(0, min(actual_interval[1], predicted_interval[1]) - max(actual_interval[0], predicted_interval[0]))
    union = max(actual_interval[1], predicted_interval[1]) - min(actual_interval[0], predicted_interval[0])
    return intersection/union

def get_f1_score():
    #actual_interval = [start, end]
    #predicted_interval = [start, end]
    actual_interval = [hhmmss_to_seconds(actual_interval[0]), hhmmss_to_seconds(actual_interval[1])]
    predicted_interval = [hhmmss_to_seconds(predicted_interval[0]), hhmmss_to_seconds(predicted_interval[1])]
    intersection = max(0, min(actual_interval[1], predicted_interval[1]) - max(actual_interval[0], predicted_interval[0]))
    precision = intersection/(predicted_interval[1] - predicted_interval[0])
    recall = intersection/(actual_interval[1] - actual_interval[0])
    return 2*precision*recall/(precision + recall)

# I have 'n_crime' number of crimmes and for each crime I have 'n_vid' number of videos and 'n_query' number of queries
# I want to calculate the average temporal iou and f1 score for each crime corresponding to each query i.e for each crime i want 'n_query' number of average temporal iou and print in a table

 
    