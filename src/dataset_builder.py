import os
from ffmpeg_helper import *

def build_dataset(source_path, output_path, resolution, seconds, splits, extensions=['mkv', 'avi', 'mp4']):
    '''Builds a dataset in the target directory by reading all the existing movie
    files from the source directory and converting them to the specified resolution.
    
    source_path(str) -- the source path with the movie files collection
    output_path(str) -- the target directory to use to build the dataset
    resolution(int) -- the desired horizontal resolution of the exported frames
    seconds(int) -- the number of seconds to extract from each video file
    splits(int) -- the number of separate video sections to use to extract the frames
    extensions(list<str>) -- a list of video extensions to filter'''

    assert extensions           # ensure at least an extension is present
    assert resolution >= 480    # not much sense in going lower than that
    assert seconds > 1          # what's the point otherwise?
    assert splits >= 1

    i, split_seconds = 0, seconds // splits
    assert split_seconds >= 1   # edge case
    ends = tuple(['.{}'.format(extension) for extension in extensions])
    
    for subdir, _, files in os.walk(source_path):
        for f in files:
            if f.endswith(ends):
                video_path = '{}\\{}'.format(subdir, f)

                # check the video duration, skip if needed
                duration = get_video_duration(video_path)
                if duration < split_seconds:
                    continue
                
                # extract frames evenly from the specified number of video sections
                step = duration // (splits + 1)
                for chunk in range(splits):
                    extract_frames(
                        video_path, resolution, output_path,
                        step * (chunk + 1) - (split_seconds // 2), # offset to before the current chunk
                        split_seconds,
                        'v{}_s{}_'.format(i, chunk))
                i += 1
