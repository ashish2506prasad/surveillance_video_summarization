from datetime import datetime, timedelta
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
import pandas as pd
import numpy as np
import json

def total_seconds_difference(start_time, end_time):
    # Calculate the total seconds between two datetime objects
    return (end_time - start_time).total_seconds()

def generate_video_clips_info(video_path, query_path, interval, qid):
    # Extracting video information
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Reading query from the file
    with open(query_path, 'r') as query_file:
        query = query_file.read().strip()

    # Calculating clip duration in seconds
    clip_duration = 90

    # Parsing start and end times from the interval
    start_time_str, end_time_str = interval
    start_time = datetime.strptime(start_time_str, '%H:%M:%S')
    end_time = datetime.strptime(end_time_str, '%H:%M:%S')

    # Calculating total video duration
    total_duration = total_seconds_difference(start_time, end_time)

    # convert to seconds
    start_time = start_time.hour * 3600 + start_time.minute * 60 + start_time.second
    end_time = end_time.hour * 3600 + end_time.minute * 60 + end_time.second

    # Calculating the number of clips
    num_clips = int(total_duration / clip_duration)

    # Generating relevant clip information
    relevant_clip_ids = []
    clips_info = []

    if start_time % clip_duration != 0:
        start_time = start_time - start_time % clip_duration
    if end_time % clip_duration != 0:
        end_time = end_time + end_time % clip_duration

    # Convert to integer here
    if int(start_time/clip_duration) +1 == int(end_time/clip_duration) + 1:
        clips_info = [int(start_time/clip_duration) + 1]
        relevant_clip_ids = [int(start_time/clip_duration) + 1]
    else:
        clips_info = [int(start_time/clip_duration) + 1, int(end_time/clip_duration) + 1]
        relevant_clip_ids = [i for i in range(int(start_time/clip_duration) + 1, int(end_time/clip_duration) + 1)]

    # Constructing the final dictionary
    result_dict = {
        "qid": qid,  # You may adjust the query ID generation logic as needed
        "query": query,
        "duration": num_clips,
        "vid": video_name,
        "relevant_clip_ids": relevant_clip_ids,
        "relevant_windows": clips_info
    }

    return result_dict

data_list = []

root_path = r"D:\BARC\week5\test_data"
crime_list  = os.listdir(root_path)
qid = 1
for crime in crime_list:
    vid_path = os.path.join(root_path,crime,"video")
    query_path = os.path.join(root_path,crime,"query") 
    vid_list = os.listdir(vid_path)

    start_end = pd.read_csv(os.path.join(root_path,crime,"start_end.csv"))
    i = 0
    for vid in vid_list:
        video_path = os.path.join(vid_path,vid)
        interval = [start_end.iloc[i,0],start_end.iloc[i,1]]
        query = os.listdir(query_path)
        for query_file in query:
            result_dict = generate_video_clips_info(video_path, os.path.join(query_path,query_file), interval,qid)
            data_list.append(result_dict)
            qid += 1
        i += 1

output_file_path = "D:\BARC\week5\output.jasonl"
with open(output_file_path, "w") as jsonl_file:
    for data in data_list:
        jsonl_file.write(json.dumps(data) + "\n")

print(f"JSONL file saved to {output_file_path}")


