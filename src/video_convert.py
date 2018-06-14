import os
import re
from shutil import copyfile
from helpers.ffmpeg_helper import extract_all_frames, create_video
from helpers.logger import LOG, INFO, BAR, RESET_LINE
from inference import process_frames

# TODO: switch to parameters
VIDEO_PATH = r"D:\ML\th\datasets\sample.mp4"
WINDOW_SIZE = 1
extension = 'jpg'

# extract the frames and execute the inference pass
LOG('Processing {}'.format(VIDEO_PATH))
working_path, frames_path = extract_all_frames(VIDEO_PATH)
process_frames(working_path)

# sort the frames by alternating the original and the interpolated
LOG('Preparing generated frames')
frames = os.listdir(working_path)

def comparer(name):
    info = re.findall('([0-9]+)(_)?', name)[0]
    return (int(info[0]), info[1])
frames.sort(key=comparer)

# duplicate the last frame (no interpolation available)
copy_filename = 'f{}_{}'.format(re.findall('([0-9]+)', frames[-1])[0], frames[-1][-4:])
copyfile('{}\\{}'.format(working_path, frames[-1]), '{}\\{}'.format(working_path, copy_filename))
frames += [copy_filename]
INFO('{} total frame(s) to encode'.format(len(frames)))

# rename the source frames to encode
for i in range(len(frames), 0, -1):
    source = '{}\\{}'.format(working_path, frames[i - 1])
    destination = '{}\\f{:03d}.{}'.format(working_path, i, extension)
    os.rename(source, destination)

# encode the interpolated video
LOG('Encoding output video')
create_video(frames_path, VIDEO_PATH)
