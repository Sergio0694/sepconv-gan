from os.path import dirname, basename
from pathlib import Path
from subprocess import call, Popen, PIPE, STDOUT, TimeoutExpired

def extract_all_frames(video_path, output_path, x=-1, extension='jpg'):
    '''Extracts all the frames from the input video.
    
    video_path(str) -- the path to the input video to process
    output_path(str) -- the path where to save the output frames
    x(int) -- the desired horizontal resolution of the exported frames
    extension(str) -- the preferred image extension for the exported frames (jpg|png|bmp)
    '''

    assert x >= 480 or x == -1 # minimum resolution

    # setup
    output_folder = '{}\\{}_'.format(output_path, basename(video_path).split('.')[0])
    frames_formatted_path = '{}\\f%03d.{}'.format(output_folder, extension)
    Path(output_folder).mkdir(exist_ok=True)
    args = [
        'ffmpeg',
        '-i', video_path,
        '-q:v', '1',
        '-qmin', '1',
        '-qmax', '1',
        '-pix_fmt', 'rgb24',
        '-v', 'quiet',
        frames_formatted_path
    ]

    # optional rescaling
    if x != -1:
        args.insert(2, '-vf')
        args.insert(3, 'scale={}:-1'.format(x))

    # process and return the output folder path
    call(args)
    return output_folder, frames_formatted_path

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
    assert duration >= 1 # really?

    Path(output_folder).mkdir(exist_ok=True)
    try:
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
        ], timeout=10)
        return True
    except TimeoutExpired:
        return False

def get_video_duration(video_path):
    '''Returns the duration of the video specified by the given path, in seconds.
    
    video_path(str) -- the path of the video to analyze
    '''

    result = Popen(
        'ffprobe -i "{}" -show_entries format=duration -v quiet -of csv="p=0"'.format(video_path), 
        stdout=PIPE,
        stderr=STDOUT)
    output = result.communicate()
    try:
        return int(float(output[0])) # bytes from PIPE > float > round to int
    except ValueError:
        return -1

def create_video(frames_path, original_path, output_path):
    '''Creates an interpolated video from the input frames.

    frames_path(str) -- the formatted path of the folder with the source frames (returned by extract_all_frames)
    extension(str) -- the extension of the source frames
    original_path(str) -- the path of the original video
    output_path(str) -- the path of the video file to create
    '''

    call([
        'ffmpeg',
        '-loglevel', 'fatal',
        '-y',
        '-framerate', '48000/1001',
        '-start_number', '1',
        '-i', frames_path,
        '-i', original_path,
        '-c:v', 'libx265',
        '-crf', '17',
        '-r', '48000/1001',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'copy',
        '-strict', 'experimental',
        '-shortest',
        output_path
    ])
    return output_path
