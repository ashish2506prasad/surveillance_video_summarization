import os
import pdb
import time
import torch
import gradio as gr
import numpy as np
import argparse
import subprocess
import pathlib as Path
from run_on_video import clip, vid2clip, txt2clip
import dataset_generation
import evaluation_metrics
import pandas as pd

parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', type=str, default='/content/UniVTG/tmp')
parser.add_argument('--resume', type=str, default='./results/omni/model_best.ckpt')
parser.add_argument("--gpu_id", type=int, default=2)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

#################################
model_version = "ViT-B/32"
output_feat_size = 512
clip_len = 2
overwrite = True
num_decoding_thread = 4
half_precision = False

clip_model, _ = clip.load(model_version, device=args.gpu_id, jit=False)

import logging
import torch.backends.cudnn as cudnn
from main.config import TestOptions, setup_model
from utils.basic_utils import l2_normalize_np_array

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)

def load_model():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse(args)
    # pdb.set_trace()
    cudnn.benchmark = True
    cudnn.deterministic = False

    if opt.lr_warmup > 0:
        total_steps = opt.n_epoch
        warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        opt.lr_warmup = [warmup_steps, total_steps]

    model, criterion, _, _ = setup_model(opt)
    return model

vtg_model = load_model()

def convert_to_hms(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def load_data(save_dir):
    vid = np.load(os.path.join(save_dir, 'vid.npz'))['features'].astype(np.float32)
    txt = np.load(os.path.join(save_dir, 'txt.npz'))['features'].astype(np.float32)

    vid = torch.from_numpy(l2_normalize_np_array(vid))
    txt = torch.from_numpy(l2_normalize_np_array(txt))
    clip_len = 2
    ctx_l = vid.shape[0]

    timestamp =  ( (torch.arange(0, ctx_l) + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

    if True:
        tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        tef_ed = tef_st + 1.0 / ctx_l
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        vid = torch.cat([vid, tef], dim=1)  # (Lv, Dv+2)

    src_vid = vid.unsqueeze(0).cuda()
    src_txt = txt.unsqueeze(0).cuda()
    src_vid_mask = torch.ones(src_vid.shape[0], src_vid.shape[1]).cuda()
    src_txt_mask = torch.ones(src_txt.shape[0], src_txt.shape[1]).cuda()

    return src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l

def forward(model, save_dir, query, actual_interval):
    src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l = load_data(save_dir)
    src_vid = src_vid.cuda(args.gpu_id)
    src_txt = src_txt.cuda(args.gpu_id)
    src_vid_mask = src_vid_mask.cuda(args.gpu_id)
    src_txt_mask = src_txt_mask.cuda(args.gpu_id)

    model.eval()
    with torch.no_grad():
        output = model(src_vid=src_vid, src_txt=src_txt, src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)
    
    # prepare the model prediction
    pred_logits = output['pred_logits'][0].cpu()
    pred_spans = output['pred_spans'][0].cpu()
    pred_saliency = output['saliency_scores'].cpu()

    # prepare the model prediction
    pred_windows = (pred_spans + timestamp) * ctx_l * clip_len
    pred_confidence = pred_logits
    
    pred_windows_clone = torch.argsort(pred_confidence, dim=0)
    # grounding
    top1_window = pred_windows[pred_windows_clone[0].item()].tolist()
    top2_window = pred_windows[pred_windows_clone[1].item()].tolist()
    top3_window = pred_windows[pred_windows_clone[2].item()].tolist()
    top4_window = pred_windows[pred_windows_clone[3].item()].tolist()
    top5_window = pred_windows[pred_windows_clone[4].item()].tolist()
    
    top5_values, top5_indices = torch.topk(pred_confidence.flatten(), k=5)
    # print("top 5 probabilities: ", top5_values)
    # print("top 5 indices of the probabilities: ", top5_indices)


    top5_windows = pred_windows[top5_indices].tolist()
    
    # print(f"The video duration is {convert_to_hms(src_vid.shape[1]*clip_len)}.")
    q_response = f"For query: {query}"

    mr_res1 =  " - ".join([convert_to_hms(int(i)) for i in top1_window])
    mr_response1 = f"The Top-1 interval is: {mr_res1}"
    mr_res2 =  " - ".join([convert_to_hms(int(i)) for i in top2_window])
    mr_response2 = f"The Top-2 interval is: {mr_res2}"
    mr_res3 =  " - ".join([convert_to_hms(int(i)) for i in top3_window])
    mr_response3 = f"The Top-3 interval is: {mr_res3}"
    mr_res4 =  " - ".join([convert_to_hms(int(i)) for i in top4_window])
    mr_response4 = f"The Top-4 interval is: {mr_res4}"
    mr_res5 =  " - ".join([convert_to_hms(int(i)) for i in top5_window])
    mr_response5 = f"The Top-5 interval is: {mr_res5}"
    
    hl_res = convert_to_hms(torch.argmax(pred_saliency) * clip_len)
    hl_response = f"The Top-1 highlight is: {hl_res}"

    # evaluate the model on f1 score and iou metric
    top5_video_iou_score = []
    top5_video_f1_score = []

    for i in range(1,6):
        start,end = f"m_res{i}".split(" - ")
        predicted_interval = [start,end]
        iou_score = evaluation_metrics.get_temporal_iou(actual_interval, predicted_interval)
        f1_score = evaluation_metrics.get_f1_score(actual_interval,predicted_interval)
        top5_video_iou_score.append(iou_score)
        top5_video_f1_score.append(f1_score)

    return '\n'.join([q_response, mr_response1, mr_response2, mr_response3, mr_response4, mr_response5]), top5_video_iou_score, top5_video_f1_score


def extract_vid(vid_path, state):
    history = state['messages']
    # list_of_crimes = os.listdir(vid_path)
    vid_features = vid2clip(clip_model, vid_path, args.save_dir)
    history.append({"role": "user", "content": "Finish extracting video features."}) 
    history.append({"role": "system", "content": "Please Enter the text query."}) 
    chat_messages = [(history[i]['content'], history[i+1]['content']) for i in range(0, len(history),2)]
    return '', chat_messages, state

def extract_txt(txt):
    txt_features = txt2clip(clip_model, txt, args.save_dir)
    return

def download_video(url, save_dir='./examples', size=768):
    save_path = f'{save_dir}/{url}.mp4'
    cmd = f'yt-dlp -S ext:mp4:m4a --throttled-rate 5M -f "best[width<={size}][height<={size}]" --output {save_path} --merge-output-format mp4 https://www.youtube.com/embed/{url}'
    if not os.path.exists(save_path):
        try:
            subprocess.call(cmd, shell=True)
        except:
            return None
    return save_path

def get_empty_state():
    return {"total_tokens": 0, "messages": []}


# save the video in the directory: ./data/crime/video/video.mp4
# save the text in the directory: ./data/crime/query/query.txt 
# save the csv file in the directory: ./data/crime/start_end.csv
def submit_message(input_value, state):   # input_value : ./data
    print("input message: ", input_value)
    history = state['messages']
    top5_video_iou_score = []
    top5_video_f1_score = []

    if not input_value:
        return gr.update(value=''), [(history[i]['content'], history[i+1]['content']) for i in range(0, len(history)-1, 2)], state

    # Check if the input is a directory
    if os.path.isdir(input_value):
        # Process all text files in the directory
        print(os.listdir(input_value).remove(".ipynb_checkpoints"))
        crime_list = os.listdir(input_value)
        crime_list.remove(".ipynb_checkpoints") if ".ipynb_checkpoints" in crime_list else None

        for crime in crime_list: # file_patth are the crime in this case around 13 crimes
            print()
            print("Processing the anomaly video: ", crime)
            start_end_df = pd.read_csv(os.path.join(input_value,crime,"start_end.csv"))
            i=1
            history.append({"role": "system", "content": f"\nProcessing Crime: {crime}"})
            
            # this for loop is for the video in the crime
            for video_path in os.listdir(os.path.join(input_value,crime,"video")):
                video = os.path.join(input_value,crime,"video",video_path)
                query_path = os.path.join(input_value,crime,"query")
                print("Processing the anomaly video index: ", i)
                history.append({"role": "user", "content": f"Loaded video from file: {video}"})
                actual_interval  = start_end_df['start'][i-1]  # actual intervla of a particular video
                # process each video for the crime
                for text_file in os.listdir(query_path): # text_file: ./test_video/crime/query
                    query_file = os.path.join(query_path,text_file)
                    with open(query_file, 'r') as file:
                        query = file.read().strip()
                        history.append({"role": "user", "content": f"Loaded query from file: {query_file}"})
                        try:
                            extract_txt(query)
                            extract_vid(video, state)  

                            answer, iou_vid, f1_score_vid  = forward(vtg_model, args.save_dir, query, actual_interval=actual_interval)  # for a particular query
                            history.append({"role": "system", "content": answer})
                            top5_video_f1_score.append(f1_score_vid)
                            top5_video_iou_score.append(iou_vid)
                        except Exception as e:
                            history.append({"role": "system", "content": f"Error processing query from file {query_file}: {e}"})
                print("Finished processing the anomaly video index: ", i)
                i = 1+1  
            print("Finished processing the anomaly video: ", crime)
            print("Average iou score for each query: ", np.mean(np.array(top5_video_iou_score), axis = 0))
            print("Average f1 score for each query: ", np.mean(np.array(top5_video_f1_score), axis = 0))
        
        # number_of_vis_processed = top5_video_f1_score.shape[0]
        print("\n")
        print("Number of videos processed: ", top5_video_f1_score.shape[0])
        
        print("Average iou score: ", np.mean(np.array(top5_video_iou_score).reshape(1,-1)))
        print("Average f1 score: ", np.mean(np.array(top5_video_f1_score).reshape(1,-1)))

    # if directory is not given, process the single query for a single video
    else:
        # Process the single query
        prompt_msg = {"role": "user", "content": input_value}
        history.append(prompt_msg)
        try:
            extract_txt(input_value)
            answer = forward(vtg_model, args.save_dir, input_value)
            history.append({"role": "system", "content": answer})
        except Exception as e:
            history.append({"role": "system", "content": f"Error: {e}"})

    chat_messages = [(history[i]['content'], history[i+1]['content']) for i in range(0, len(history)-1, 2)]
    return '', chat_messages, state


def clear_conversation():
    return gr.update(value=None, visible=True), gr.update(value=None, interactive=True), None, gr.update(value=None, visible=True), get_empty_state()


def subvid_fn(vid):
    save_path = download_video(vid)
    return gr.update(value=save_path)


css = """
      #col-container {max-width: 80%; margin-left: auto; margin-right: auto;}
      #video_inp {min-height: 100px}
      #chatbox {min-height: 100px;}
      #header {text-align: center;}
      #hint {font-size: 1.0em; padding: 0.5em; margin: 0;}
      .message { font-size: 1.2em; }
      """

with gr.Blocks(css=css) as demo:
    
    state = gr.State(get_empty_state())


    with gr.Column(elem_id="col-container"):
        gr.Markdown("""## 🤖️ UniVTG: Towards Unified Video-Language Temporal Grounding
                    Given a video and text query, return relevant window and highlight.
                    https://github.com/showlab/UniVTG/""",
                    elem_id="header")

        with gr.Row():
            with gr.Column():
                video_inp = gr.Video(label="video_input")
                gr.Markdown("👋 **Step1**: Select a video in Examples (bottom) or input youtube video_id in this textbox, *e.g.* *G7zJK6lcbyU* for https://www.youtube.com/watch?v=G7zJK6lcbyU and enter query in the textbox provided", elem_id="hint")
                with gr.Row():
                    video_id = gr.Textbox(value="", placeholder="Youtube video url", show_label=False)
                    vidsub_btn = gr.Button("(Optional) Submit Youtube id")

            with gr.Column():
                vid_ext = gr.Button("Step2: Extract video feature, may takes a while")
                # vlog_outp = gr.Textbox(label="Document output", lines=40)
                total_tokens_str = gr.Markdown(elem_id="total_tokens_str")
                
                chatbot = gr.Chatbot(elem_id="chatbox")
                input_message = gr.Textbox(show_label=False, placeholder="Or Enter path to video and query and press enter", visible=True).style(container=False)
                btn_submit = gr.Button("Step3: Enter your path to text query")
                btn_clear_conversation = gr.Button("🔃 Clear")

        examples = gr.Examples(
            examples=[
                # ["./examples/youtube.mp4"], 
                ["/content/UniVTG/examples/charades.mp4"],
                # ["/content/UniVTG/examples/michael_phelps_last_olympic_race.mp4"], 
                # ["./examples/ego4d.mp4"],
            ],
            inputs=[video_inp],
        )

    gr.HTML('''<br><br><br><center>You can duplicate this Space to skip the queue:<a href="https://huggingface.co/spaces/anzorq/chatgpt-demo?duplicate=true"><img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a><br></center>''')

    btn_submit.click(submit_message, [input_message, state], [input_message, chatbot])
    input_message.submit(submit_message, [input_message, state], [input_message, chatbot])
    # btn_clear_conversation.click(clear_conversation, [], [input_message, video_inp, chatbot, vlog_outp, state])
    btn_clear_conversation.click(clear_conversation, [], [input_message, video_inp, chatbot, state])
    vid_ext.click(extract_vid, [video_inp, state], [input_message, chatbot])
    vidsub_btn.click(subvid_fn, [video_id], [video_inp])

    demo.load(queur=False)


demo.queue(concurrency_count=10)
demo.launch(height='800px', server_port=2253, debug=True, share=True)