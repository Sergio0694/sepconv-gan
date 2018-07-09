import os
from pathlib import Path
from subprocess import call, Popen, PIPE, STDOUT, TimeoutExpired

def extract_frames(video_path, output_folder, scale=None, start=0, duration=60, suffix='', extension='.jpg', timeout=None):
    '''Exports a series of frames from the input video to the specified folder.

    video_path(str) -- the path to the input video
    output_folder(str) -- the path of the desired output folder for the video frames
    scale(list<int>) -- the optional desired resolution of the exported frames
    start(int) -- the export start time, in seconds
    duration(int) -- the number of seconds to export
    suffix(str) -- an identifier for the exported frames
    extension(str) -- the preferred image extension for the exported frames (jpg|png|bmp)
    timeout(int) -- optional timeout for the operation
    '''
    
    assert video_path is not None and output_folder is not None
    assert timeout is None or timeout >= 10
    assert start >= 0
    assert duration >= 1 # really?
    assert scale is None or \
            (len(scale) == 2 and (scale[0] >= 240 or scale[0] == -1) \
            and (scale[1] >= 240 or scale[1] == -1) and not (scale[0] == -1 and scale[1] == -1))
    assert timeout is None or timeout >= 1

    Path(output_folder).mkdir(exist_ok=True)
    args = (
        ['ffmpeg'] +
        (['-ss', str(start)] if start > 0 else []) + # optional start time
        ['-i', '"{}"'.format(video_path)] +
        ['-to', str(duration)] + # -ss resets the timestep to target start time
        (['-vf', 'scale={}:{}'.format(scale[0], scale[1])] if scale is not None else []) + # optional rescaling
        ['-q:v', '1'] +
        ['-qmin', '1'] +
        ['-qmax', '1'] +
        ['-v', 'quiet'] +
        ['"{}"'.format(os.path.join(output_folder, '{}%05d{}'.format(suffix, extension)))])

    if timeout:
        try:
            Popen(' '.join(args), shell=True, stdout=PIPE, stderr=STDOUT).communicate(timeout=timeout)
            return True
        except TimeoutExpired:
            return False
    else:
        Popen(' '.join(args), shell=True, stdout=PIPE, stderr=STDOUT).communicate()
        return True

def get_video_duration(video_path):
    '''Returns the duration of the video specified by the given path, in seconds.
    
    video_path(str) -- the path of the video to analyze
    '''

    result = Popen(
        'ffprobe -i "{}" -show_entries format=duration -v quiet -of csv="p=0"'.format(video_path),
        shell=True,
        stdout=PIPE,
        stderr=STDOUT)
    output = result.communicate()
    try:
        return int(float(output[0])) # bytes from PIPE > float > round to int
    except ValueError:
        return -1
