# surveillance_video_summarization

## Link to data set:
The UCF-Crime dataset is a large-scale dataset of 128 hours of videos. It consists of 1900 long and untrimmed real-world surveillance videos, with 13 realistic anomalies including Abuse, Arrest, Arson, Assault, Road Accident, Burglary, Explosion, Fighting, Robbery, Shooting, Stealing, Shoplifting, and Vandalism. These anomalies are selected because they have a significant impact on public safety.
https://www.kaggle.com/datasets/minhajuddinmeraj/anomalydetectiondatasetucf?resource=download-directory&select=Anomaly-Videos-Part-1

## Link to data set and summary
#### V1:
https://drive.google.com/drive/folders/13tg5TGEG6PCgGidF3N08cq7cjUDMFsI7?usp=sharing
#### V2:
https://drive.google.com/drive/folders/1lvAA97ApY2Zc4Ie5xNEkWIfolIUr-ElJ?usp=sharing

## Checkpoints
download the checkpoints from the link given below and save it in UniVTG/results/omni
#### link: https://drive.google.com/drive/folders/1l6RyjGuqkzfZryCC6xwTZsvjWaIMVxIO?usp=sharing

## Manual ispection
save the videos that needs to be inspected in /UniVTG/examples and make changes in the main_gradio.py file accordingly

## Data generation
run the file Data_generation.py. It requires you to save the videos in the following format
./Anomaly_videos/Arson/Video_1.mp4
The queries have to be manually generated
The script returns a csv file containing ground truth temporal labels

## For evaluation of the model
store the data in the following format:
./Arson/video/video_1.mp4    (Arson is just one example of a category, you can have as many videos as you want)
./Arson/query/query_1.txt    (as many queries as you want)
./Arson/start_end.csv        (a csv file containing start and end time)



