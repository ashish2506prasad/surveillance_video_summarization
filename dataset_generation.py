import os
import numpy as np
import cv2

###########################################################################################
# calculate the length of a video in seconds
def get_length(file_directory):
    cap = cv2.VideoCapture(file_directory)
    fps = cap.get(cv2.CAP_PROP_FPS)  # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    return duration

def seconds_to_hhmmss(seconds):
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

###########################################################################################
def resize_and_concatenate_videos(video_paths, output_path, new_width, new_height):
    # Check if the output directory exists, create it if not
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize an empty list to store the resized video frames
    resized_frames = []

    # Iterate over each video path
    for video_path in video_paths:
        # Read the video file
        cap = cv2.VideoCapture(video_path)

        # Resize each frame to the new width and height
        resized_frames_per_video = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            resized_frame = cv2.resize(frame, (new_width, new_height))
            resized_frames_per_video.append(resized_frame)

        # Close the video capture
        cap.release()

        # Append the resized frames to the main list
        resized_frames.append(resized_frames_per_video)

    # Concatenate the resized frames in the temporal domain
    concatenated_frames = np.concatenate(resized_frames, axis=0)

    # Write the concatenated frames to a new video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width * len(video_paths), new_height))

    for frame in concatenated_frames:
        out.write(frame)

    # Release the video writer
    out.release()

###########################################################################################
# define hyperparameters
n = 2  # number of videos to be stiched
num_crimes = 1
vids_per_crime = 3
new_width = 64
new_height = 64

root_path = "D:\BARC\week_4\Anomaly Videos"
save_dir = 'D:\BARC\week_4\data_set'
os.makedirs(save_dir, exist_ok=True)
###########################################################################################

path_to_normal_videos = os.path.join(root_path, "normal")
# path_to_crime_videos = os.path.join(root_path, "crime")

crime = os.listdir(root_path)
crime.remove("normal")

for i in range(num_crimes):
    anomaly = crime[i]
    for j in range(vids_per_crime):
        # get a random video from each crime
        random_vid = np.random.choice(os.listdir(os.path.join(root_path, anomaly)), 1, replace=False)[0]
        # choose 4 random videos from normal videos
        random_normal_vids = np.random.choice(os.listdir(path_to_normal_videos), n - 1, replace=False)
        idx = np.random.randint(0, n)
        # insert the anomaly video at a random index
        vids = list(random_normal_vids)
        vids.insert(idx, random_vid)
        # calculate the length of all the videos in the list vids
        k = 0
        start = 0
        while k < idx:
            length = get_length(os.path.join(path_to_normal_videos, vids[k]))
            start += length
            k += 1
        end = start + get_length(os.path.join(root_path, anomaly, random_vid))

        # make a list of directories in the order of the videos
        video_paths = [os.path.join((path_to_normal_videos if k != idx else os.path.join(root_path, anomaly)),
                                    video) for k, video in enumerate(vids)]
        # concatenate the videos
        output_vid_path = os.path.join(save_dir, anomaly, f"concatenated_{i}_{j}.mp4")
        resize_and_concatenate_videos(video_paths, output_vid_path, new_width, new_height)
        # save the start and end time of the anomaly of each video in a csv file in the same csv file
        start_time_str = seconds_to_hhmmss(start)
        end_time_str = seconds_to_hhmmss(end)
        print(start_time_str, end_time_str)
        output_path = os.path.join(save_dir, anomaly)
        csv_output_path = os.path.join(output_path, "start_end.csv")
        # if the directory does not exist make it
        # Check if the file already exists
        if os.path.exists(csv_output_path):
            with open(csv_output_path, "a") as f:
                f.write(f"{start_time_str},{end_time_str}\n")
        else:
            # if the directory does not exist make it
            os.makedirs(os.path.dirname(csv_output_path), exist_ok=True)
            with open(csv_output_path, "w") as f:
                f.write("start,end\n")
                f.write(f"{start_time_str},{end_time_str}\n")

print("Done")

