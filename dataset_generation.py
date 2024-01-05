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

###########################################################################################
def resize_and_concatenate_videos(video_directories, output_path):
    # Check if the output directory exists, create it if not
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Initialize an empty list to store the resized video frames
    resized_frames = []

    # Iterate over each video directory
    for video_dir in video_directories:
        # Get the list of video files in the directory
        video_files = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.avi', '.mkv'))]

        # Iterate over each video file in the directory
        for video_file in video_files:
            # Read the video file
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            # Get video properties
            width = int(cap.get(3))
            height = int(cap.get(4))
            # Resize each frame to a new width and height
            new_width = 640
            new_height = 480
            new_size = (new_width, new_height)

            resized_frames_per_video = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                resized_frame = cv2.resize(frame, new_size)
                resized_frames_per_video.append(resized_frame)
            # Close the video capture
            cap.release()
            # Append the resized frames to the main list
            resized_frames.append(resized_frames_per_video)

    # Concatenate the resized frames in the temporal domain
    concatenated_frames = np.concatenate(resized_frames, axis=1)

    # Write the concatenated frames to a new video file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (new_width * len(video_directories), new_height))

    for frame in concatenated_frames:
        out.write(frame)

    # Release the video writer
    out.release()

###########################################################################################
# define hyperparameters
n = 5 # number of videos to be stiched
num_crimes = 3
vids_per_crime = 3 

root_path = "path/to/vids"
###########################################################################################

path_to_normal_videos = os.path.join(root_path, "normal")
# path_to_crime_videos = os.path.join(root_path, "crime")

crime = os.listdir(root_path).remove("normal")  # around 13 crimes

def make_stiched_videos():

    for i in range(num_crimes):
        anomaly = crime[i]
        for _ in range(vids_per_crime):
            # get a random video from each crime
            random_vid = np.random.choice(os.listdir(os.pat.join(root_path, anomaly)), 1, replace=False) 
            # choose 4 random videos from normal videos
            random_normal_vids = np.random.choice(os.listdir(path_to_normal_videos), n-1, replace=False) 

            idx = np.randint(0,n)
            # insert the anomaly video at a random index
            vids = random_normal_vids.insert(idx, random_vid)
            # calculate the length of all the videos in the list vids
            j=0
            start = 0
            while j<idx:
                length = get_length(os.path.join(path_to_normal_videos, vids[j]))
                start = start + length
                j+=1
            end = start + get_length(os.path.join(root_path, anomaly, random_vid))

            # make a list of directories in the order of the videos
            video_paths = [os.path.join((path_to_normal_videos if i != idx else os.path.join(root_path,anomaly)), vids[i]) for i in range(n)]
            # concatenate the videos
            output_path = os.path.join(root_path, "concatenated", anomaly, str(i))
            resize_and_concatenate_videos(video_paths, output_path)
            # save the video
            save_vid(output_path, os.path.join(root_path, "concatenated", anomaly, str(i), "concatenated.mp4"))
    return start, end
        
    
        



    

