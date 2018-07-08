from collections import defaultdict
from multiprocessing import cpu_count, Process, Queue
import os
from shutil import rmtree
import cv2
import numpy as np
from helpers.ffmpeg_helper import *
from helpers.logger import LOG, INFO, BAR, RESET_LINE
from __MACRO__ import *

SUPPORTED_VIDEO_FORMATS = ('.mkv', '.avi', '.mp4')

def load_files(source_paths):
    '''Loads the list of all existing video files with the supported format from the input source directories.

    source_paths(list<str>) -- the list of source directories
    '''

    source_files = []
    for source_path in source_paths:
        for subdir, _, files in os.walk(source_path):
            for f in files:
                if f.endswith(SUPPORTED_VIDEO_FORMATS):
                    source_files += [os.path.join(subdir, f)]
    return source_files

def process_video_file(queue, cpu_id, output_path, seconds, splits, min_duration, resolution, encoding, timeout):
    '''Processes a video file in the background.'''

    # output folder for the current process
    target_path = os.path.join(output_path, '_{}'.format(cpu_id))
    os.mkdir(target_path)

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
        failed = None
        aborted = False
        skipped = 0
        for chunk in range(splits):
            if not extract_frames(
                video_path, target_path, None if resolution is None else (resolution, -1),
                step * (chunk + 1) - (seconds // 2), # offset to before the current chunk
                seconds,
                'v{}-s{}_'.format(i, chunk), encoding, timeout):
                if failed == chunk - 1:
                    aborted = True
                    break
                failed = chunk
                skipped += 1
        if failed is None:
            INFO('{} OK [_{}][{}]'.format(video_path, cpu_id, i))
        elif aborted:
            INFO('{} FAILED at {} [_{}][{}]'.format(video_path, failed, cpu_id, i))
        else:
            INFO('{} OK with {} skipped [_{}][{}]'.format(video_path, skipped, cpu_id, i))

def preprocess_frames(frames_folder, extension, min_variance, min_diff_threshold, max_diff_threshold, max_length, color):

    # list the frames and build a mapping for each video chunk
    frames = os.listdir(frames_folder)
    if not frames:
        return
    mapping = defaultdict(list)
    for frame in frames:
        key = frame.split('_')[0]
        mapping[key] += [frame]

    for key in mapping.keys():
        chunk = mapping[key]
        chunk.sort()

        # load the images in the current video chunk
        data_map = {
            frame: cv2.imread(os.path.join(frames_folder, frame))
            for frame in chunk
        }
        size = np.prod(next(iter(data_map.values())).shape) # size of the first frame in the current sequence
        assert size > 0

        # calculate the error between each consecutive pair of frames
        errors_map = {
            pair[0]: np.sum((data_map[pair[0]] - data_map[pair[1]]) ** 2, dtype=np.float32) / size
            for pair in zip(chunk, chunk[1:])
        }

        # split the frames into valid subsequences
        splits = defaultdict(list)
        i = 0
        splits[i] = [chunk[0]] # base case
        for j in range(1, len(chunk)):
            if errors_map[chunk[j - 1]] < min_diff_threshold or errors_map[chunk[j - 1]] > max_diff_threshold:
                i += 1
            else:
                mean, var = cv2.meanStdDev(data_map[chunk[j]])
                if np.sum(var) / 3 < min_variance or \
                    color and np.all(np.isclose(mean, mean[0], atol=1.0)) and np.all(np.isclose(var, var[0], atol=1.0)):
                    i += 1
            if len(splits[i]) < max_length:
                splits[i] += [chunk[j]]
        
        # rename the valid sequences, delete the other frames
        root_dir = os.path.dirname(frames_folder)
        for b, split_key in enumerate(splits):
            if len(splits[split_key]) >= 3:
                for s, frame in enumerate(splits[split_key]):
                    os.rename(os.path.join(frames_folder, frame), os.path.join(root_dir, '{}-b{}_{}{}'.format(key, b, s, extension)))

def process_batch(
    n, video_files, output_path, seconds, splits, resolution, min_duration, extension, 
    min_variance, min_diff_threshold, max_diff_threshold, max_length, color,
    timeout):

    # setup the workers
    queue = Queue()
    processes = [
        Process(target=process_video_file, args=[queue, cpu_id, output_path, seconds, splits, min_duration, resolution, extension, timeout])
        for cpu_id in range(cpu_count())
    ]
    for process in processes:
        process.start()
    LOG('Workers started')

    # process the source video files
    for v, video_file in enumerate(video_files):
        queue.put((n + v, video_file))
    LOG('Queue ready')

    # wait for completion
    for _ in range(cpu_count()):
        queue.put(None)
    for process in processes:
        process.join()
    queue.close()
    LOG('Frames extraction completed')

    # preprocess the extracted frames
    subdirs = ['_{}'.format(cpu_id) for cpu_id in range(cpu_count())]
    processes = [
        Process(target=preprocess_frames, args=[os.path.join(output_path, subdir), extension, min_variance, min_diff_threshold, max_diff_threshold, max_length, color])
        for subdir in subdirs
    ]
    for process in processes:
        process.start()
    LOG('Preprocessing workers started')
    for process in processes:
        process.join()

    # cleanup
    LOG('Cleanup')
    for subdir in subdirs:
        rmtree(os.path.join(output_path, subdir))

def build_dataset(
    source_paths, output_path, seconds, splits, resolution, extension, 
    min_variance, min_diff_threshold, max_diff_threshold, max_length, color,
    timeout, batch_size):
    '''Builds a dataset in the target directory by reading all the existing movie
    files from the source directory and converting them to the specified resolution.
    
    source_paths(list<str>) -- the list of source paths with the movie files collection
    output_path(str) -- the target directory to use to build the dataset
    seconds(int) -- the duration in seconds of each extracted video clip
    splits(int) -- the number of separate video sections to use to extract the frames
    resolution(int) -- the desired horizontal resolution of the exported frames
    extension(str) -- the extension (and file format) for the output frames
    min_variance(int) -- the minimum variance value for a valid video frame
    min_diff_threshold(int) -- the minimum difference between consecutive video frames
    max_diff_threshold(int) -- the maximum difference between consecutive video frames
    max_length(int) -- the maximum length of a series of consecutive frames
    color(bool) -- indicates whether or not to filter out grayscale images
    timeout(int) -- timeout for the frames extraction operation
    batch_size(int) -- the size of each processing batch
    '''

    assert seconds >= 1                             # what's the point otherwise?
    assert splits >= 1
    assert resolution is None or resolution >= 240  # ensure minimum resolution

    min_duration = seconds * splits * 4 # some margin just in case

    # load the list of video files
    LOG('Loading source files')
    video_files = load_files(source_paths)
    LOG('{} video file(s) to process'.format(len(video_files)))

    def batch(l, n):
        '''A small function that batches the input list.'''

        i = 0
        while True:
            b = l[i:i + n]
            if not b:
                break
            yield b
            i += n

    # process the source videos
    for i, chunk in enumerate(batch(video_files, batch_size)):
        process_batch(
            batch_size * i, chunk,
            output_path, seconds, splits, resolution, min_duration, extension,
            min_variance, min_diff_threshold, max_diff_threshold, max_length, color,
            timeout)
    LOG('Dataset created successfully')
