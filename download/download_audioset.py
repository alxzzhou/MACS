import os
import random

import cv2
import moviepy.editor as mp
import pandas as pd


def download(set, name, t_seg):
    path_data = os.path.join(set, "video")
    if not os.path.exists(path_data):
        os.makedirs(path_data)
    link_prefix = "https://www.youtube.com/watch?v="

    filename_full_video = os.path.join(path_data, name) + "_full_video.%(ext)s"
    filename = os.path.join(path_data, name) + ".mp4"
    link = link_prefix + name

    if os.path.exists(filename):
        print(f"{filename} already exists, skip")
        return

    t_start, t_end = t_seg
    print("download the whole video for: [%s] - [%s]" % (set, name))
    os.system(f'yt-dlp --ignore-config {link} -o {filename_full_video} ')

    t_dur = t_end - t_start
    filename_full_video = filename_full_video[:-8]
    if os.path.exists(filename_full_video + '.mkv'):
        filename_full_video += '.mkv'
    elif os.path.exists(filename_full_video + '.webm'):
        filename_full_video += '.webm'
    else:
        filename_full_video += '.mp4'
    print(filename_full_video)
    print("trim the video to [%.1f-%.1f]" % (t_start, t_end))
    os.system(f'ffmpeg -i {filename_full_video} -ss {t_start} -t {t_dur}' +
              f' -vcodec libx264 -acodec aac -strict -2 -y {filename}' +
              f' -loglevel -8')
    if os.path.exists(filename_full_video): os.remove(filename_full_video)

    if os.path.exists(filename):
        try:
            # cap = cv2.VideoCapture(filename)
            # frame_n = random.randint(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1)
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
            # _, frame = cap.read()
            # cap.release()
            # cv2.imwrite(os.path.join(set, 'image', name) + '.jpg', frame)
            mp.VideoFileClip(filename).audio.write_audiofile(os.path.join(set, 'audio', name) + '.wav', fps=16000)
        except:
            print('error happened')
    print("finish the video as: " + filename)


segment = './eval_segments.csv'
segment = pd.read_csv(segment, header=2, sep=',')

output_dir = "/path/to/download"
for _, it in segment.iterrows():
    ytid, start, end = it['YTID'], it['start_seconds'], it['end_seconds']
    download(output_dir, ytid, (start, end))
