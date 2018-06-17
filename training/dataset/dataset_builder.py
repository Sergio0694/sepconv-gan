import os
from helpers.ffmpeg_helper import *
from helpers.logger import LOG, INFO, BAR, RESET_LINE
from __MACRO__ import *

def build_dataset(source_path, output_path, seconds, splits, x, y=-1, encoding='jpg', inputs=['mkv', 'avi', 'mp4']):
    '''Builds a dataset in the target directory by reading all the existing movie
    files from the source directory and converting them to the specified resolution.
    
    source_path(str) -- the source path with the movie files collection
    output_path(str) -- the target directory to use to build the dataset
    resolution(int) -- the desired horizontal resolution of the exported frames
    seconds(int) -- the number of seconds to extract from each video file
    splits(int) -- the number of separate video sections to use to extract the frames
    x(int) -- the desired horizontal resolution of the exported frames
    y(int) -- the desired vertical resolution of the exported frames
    encoding(str) -- the extension (and file format) for the output frames
    inputs(list<str>) -- a list of video extensions to filter'''

    assert inputs                               # ensure at least an extension is present
    assert encoding in ['jpg', 'bmp', 'png']    # ensure valid output format
    assert x >= 480                             # not much sense in going lower than that
    assert y >= 270 or y == -1                  # minimum resolution
    assert seconds > 1                          # what's the point otherwise?
    assert splits >= 1

    i, split_seconds, min_duration = 0, seconds // splits, seconds * (splits + 1)
    assert split_seconds >= 1   # edge case
    ends = tuple(['.{}'.format(extension) for extension in inputs])
    
    for subdir, _, files in os.walk(source_path):
        for f in files:
            if f.endswith(ends):
                video_path = '{}\\{}'.format(subdir, f)
                if VERBOSE_MODE:
                    LOG(video_path)

                # check the video duration, skip if needed
                duration = get_video_duration(video_path)
                if duration < min_duration:
                    if VERBOSE_MODE:
                        INFO('Video too short: {}s'.format(duration))
                    continue

                # extract frames evenly from the specified number of video sections
                step = duration // (splits + 1)
                for chunk in range(splits):
                    if VERBOSE_MODE:
                        BAR(chunk, splits)
                    if not extract_frames( #TODO: fix this (broken by refactoring)
                            video_path, output_path, x, y,
                            step * (chunk + 1) - (split_seconds // 2), # offset to before the current chunk
                            split_seconds,
                            'v{}_s{}_'.format(i, chunk), encoding):
                        RESET_LINE()
                        INFO('Failed at chunk {}'.format(chunk))
                        break
                RESET_LINE()
                i += 1