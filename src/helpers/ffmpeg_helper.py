from pathlib import Path
from subprocess import call, Popen, PIPE, STDOUT

def extract_frames(video_path, output_folder, x, y=-1, start=0, duration=60, suffix='', extension='jpg'):
    '''Exports a series of frames from the input video to the specified folder.

    video_path(str) -- the path to the input video
    output_folder(str) -- the path of the desired output folder for the video frames
    x(int) -- the desired horizontal resolution of the exported frames
    y(int) -- the desired vertical resolution of the exported frames
    start(int) -- the export start time, in seconds
    duration(int) -- the number of seconds to export
    suffix(str) -- an identifier for the exported frames
    extension(str) -- the preferred image extension for the exported frames (jpg|png|bmp)
    '''

    assert start > 0
    assert duration > 1 # really?

    Path(output_folder).mkdir(exist_ok=True)
    call([
        'ffmpeg',
        '-ss', str(start),
        '-t', str(duration),
        '-i', video_path,
        '-vf', 'scale={}:{}'.format(x, y),
        '-q:v', '1',
        '-qmin', '1',
        '-qmax', '1',
        '-pix_fmt', 'rgb24',
        '-v', 'quiet',
        '{}\\{}%03d.{}'.format(output_folder, suffix, extension)
    ])

def get_video_duration(video_path):
    '''Returns the duration of the video specified by the given path, in seconds
    
    video_path(str) -- the path of the video to analyze
    '''

    result = Popen(
        'ffprobe -i "{}" -show_entries format=duration -v quiet -of csv="p=0"'.format(video_path), 
        stdout=PIPE,
        stderr=STDOUT)
    output = result.communicate()
    return int(float(output[0])) # bytes from PIPE > float > round to int
