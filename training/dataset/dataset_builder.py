from multiprocessing import cpu_count, Process, Queue
import os
from helpers.ffmpeg_helper import *
from helpers.logger import LOG, INFO, BAR, RESET_LINE
from __MACRO__ import *

SUPPORTED_VIDEO_FORMATS = ['mkv', 'avi', 'mp4']
SUPPORTED_VIDEO_EXTENSIONS = tuple(['.{}'.format(extension) for extension in SUPPORTED_VIDEO_FORMATS])

def load_files(source_paths):
    '''Loads the list of all existing video files with the supported format from the input source directories.

    source_paths(list<str>) -- the list of source directories
    '''

    source_files = []
    for source_path in source_paths:
        for subdir, _, files in os.walk(source_path):
            for f in files:
                if f.endswith(SUPPORTED_VIDEO_EXTENSIONS):
                    source_files += [os.path.join(subdir, f)]
    return source_files

def process_video_file(queue, cpu_id, output_path, seconds, splits, min_duration, resolution, encoding):
    '''Processes a video file in the background.'''

    # output folder for the current process
    target_path = os.mkdir(os.path.join(output_path, '_{}'.format(i)))

    while True:

        # get the current video path
        task = queue.get()
        if task is None:
            break        
        i, video_path = task[0], task[1]

        # check the video duration, skip if needed
        duration = get_video_duration(video_path)
        if duration < min_duration:
            if VERBOSE_MODE:
                INFO('Video too short: {}s'.format(duration))
            continue

        # extract frames evenly from the specified number of video sections
        step = duration // (splits + 1)
        for chunk in range(splits):
            if not extract_frames(
                video_path, target_path, resolution,
                step * (chunk + 1) - (seconds // 2), # offset to before the current chunk
                seconds,
                'p{}_v{}_s{}_'.format(cpu_id, i, chunk), encoding):
                INFO('{} FAIL at {}'.format(video_path, chunk))
                break
        INFO('{} OK'.format(video_path))
        

def build_dataset(source_paths, output_path, seconds, splits, resolution, encoding, min_variance, min_diff_threshold, max_diff_threshold):
    '''Builds a dataset in the target directory by reading all the existing movie
    files from the source directory and converting them to the specified resolution.
    
    source_paths(list<str>) -- the list of source paths with the movie files collection
    output_path(str) -- the target directory to use to build the dataset
    seconds(int) -- the duration in seconds of each extracted video clip
    splits(int) -- the number of separate video sections to use to extract the frames
    resolution(int) -- the desired horizontal resolution of the exported frames
    encoding(str) -- the extension (and file format) for the output frames
    min_variance(int) -- the minimum variance value for a valid video frame
    min_diff_threshold(int) -- the minimum difference between consecutive video frames
    max_diff_threshold(int) -- the maximum difference between consecutive video frames
    '''

    assert seconds > 1                              # what's the point otherwise?
    assert splits >= 1
    assert resolution is None or resolution >= 240  # ensure minimum resolution
    assert encoding in SUPPORTED_VIDEO_FORMATS      # ensure valid output format

    min_duration = seconds * splits * 4 # some margin just in case

    # load the list of video files
    video_files = load_files(source_paths)
    LOG('{} video file(s) to process'.format(len(video_files)))

    # setup the workers
    with Queue() as queue:
        processes = [
            Process(target=process_video_file, args=[queue, cpu_id, output_path, seconds, splits, min_duration, resolution, encoding])
            for cpu_id in cpu_count()
        ]
        for process in processes:
            process.start()

        # process the source video files
        for v, video_file in enumerate(video_files):
            queue.put((v, video_file))

        # wait for completion
        for _ in range(cpu_count()):
            queue.put(None)
        for process in processes:
            process.join()
