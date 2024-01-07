import os
import numpy as np
import cv2

###########################################################################################
def save_vid(file_directory, save_directory):
    new_width=224
    new_height=224

    # Read the input video file
    cap = cv2.VideoCapture(file_directory)

    # Create VideoWriter object for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = os.path.join(save_directory)
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (new_width, new_height))

    # Process each frame in the video
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Resize the frame
        resized_frame = cv2.resize(frame, (new_width, new_height))

        # Write the resized frame to the output video
        out.write(resized_frame)

    # Release the video capture and writer objects
    cap.release()
    out.release()

###########################################################################################
# calculate the length of a video in seconds
def get_length(file_directory):
    cap = cv2.VideoCapture(file_directory)
    fps = cap.get(cv2.CAP_PROP_FPS)      # OpenCV2 version 2 used "CV_CAP_PROP_FPS"
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps

    return duration

def seconds_to_hhmmss(seconds):
    hours = seconds//3600
    seconds %= 3600
    minutes = seconds//60
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
    concatenated_frames = np.concatenate(resized_frames, axis=1)

    # Write the concatenated frames to a new video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width * len(video_paths), new_height))

    for frame in concatenated_frames:
        out.write(frame)

    # Release the video writer
    out.release()

###########################################################################################
# define hyperparameters
n = 3 # number of videos to be stiched
num_crimes = 2
vids_per_crime = 1 
new_width = 640
new_height = 480

root_path = "D:\BARC\week_4\Anomaly Videos"
###########################################################################################

path_to_normal_videos = os.path.join(root_path, "normal")
# path_to_crime_videos = os.path.join(root_path, "crime")

crime = os.listdir(root_path)
crime.remove("normal")


for i in range(num_crimes):
    anomaly = crime[i]
    for _ in range(vids_per_crime):
        # get a random video from each crime
        random_vid = np.random.choice(os.listdir(os.path.join(root_path, anomaly)), 1, replace=False)[0]
        # choose 4 random videos from normal videos
        random_normal_vids = np.random.choice(os.listdir(path_to_normal_videos), n-1, replace=False)
        idx = np.random.randint(0, n)
        # insert the anomaly video at a random index
        vids = list(random_normal_vids)
        vids.insert(idx, random_vid)
        # calculate the length of all the videos in the list vids
        j = 0
        start = 0
        while j < idx:
            length = get_length(os.path.join(path_to_normal_videos, vids[j]))
            start += length
            j += 1
        end = start + get_length(os.path.join(root_path, anomaly, random_vid))

        # make a list of directories in the order of the videos
        video_paths = [os.path.join((path_to_normal_videos if i != idx else os.path.join(root_path, anomaly)), video) for i, video in enumerate(vids)]
        # concatenate the videos
        output_path = os.path.join(root_path, "concatenated", anomaly)
        resize_and_concatenate_videos(video_paths, output_path, new_width, new_height)
        # save the video
        save_vid(os.path.join(root_path, "concatenated", anomaly, "concatenated" + str(i) + ".mp4"), output_path)
        # save the start and end time of the anomaly of each video in a csv file in the same csv file
        start = seconds_to_hhmmss(start)
        end = seconds_to_hhmmss(end)
        with open(os.path.join(root_path, "concatenated", anomaly, "start_end.csv"), "w") as f:
            f.write("start,end\n")
            f.write(start + "," + end + "\n")

print("Done!")
print("path to saved videos:", os.path.join(root_path, "concatenated"))
