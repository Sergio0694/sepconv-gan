from pathlib import Path
from subprocess import call, Popen, PIPE, STDOUT, TimeoutExpired

def extract_frames(video_path, output_folder, scale=None, start=0, duration=60, suffix='', extension='jpg'):
    '''Exports a series of frames from the input video to the specified folder.

    video_path(str) -- the path to the input video
    output_folder(str) -- the path of the desired output folder for the video frames
    scale(list<int>) -- the optional desired resolution of the exported frames
    start(int) -- the export start time, in seconds
    duration(int) -- the number of seconds to export
    suffix(str) -- an identifier for the exported frames
    extension(str) -- the preferred image extension for the exported frames (jpg|png|bmp)
    '''
    
    assert start >= 0
    assert duration >= 1 # really?
    assert scale is None or \
            (len(scale) == 2 and (scale[0] >= 240 or scale[0] == -1) \
            and (scale[1] >= 240 or scale[1] == -1) and not (scale[0] == -1 and scale[1] == -1))

    Path(output_folder).mkdir(exist_ok=True)
    args = [
        'ffmpeg',
        '-i', video_path,
        '-to', str(duration), # -ss resets the timestep to target start time
        '-q:v', '1',
        '-qmin', '1',
        '-qmax', '1',
        '-pix_fmt', 'rgb24',
        '-v', 'quiet',
        '{}\\{}%03d.{}'.format(output_folder, suffix, extension)
    ]

    # optional start time
    if start > 0:
        args.insert(1, '-ss')
        args.insert(2, str(start))

    # optional rescaling
    if scale is not None:
        args.insert(4, '-vf')
        args.insert(5, 'scale={}:{}'.format(scale[0], scale[1]))

    try:
        call(args, timeout=10)
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

def create_video(frames_path, output_path, encoder='h264', crf='23', preset='normal'):
    '''Creates an interpolated video from the input frames.

    frames_path(str) -- the formatted path of the folder with the source frames (returned by extract_all_frames)
    output_path(str) -- the path of the video file to create
    encoder(str) -- the encoder to use
    crf(str) -- the CRF value for the encoder
    preset(str) -- the encoder preset to use
    '''

    call([
        'ffmpeg',
        '-y',
        '-framerate', '48000/1001',
        '-start_number', '1',
        '-i', frames_path,
        '-c:v', 'libx{}'.format(encoder[1:]),
        '-crf', crf,
        '-preset', preset,
        '-r', '48000/1001',
        '-pix_fmt', 'yuv420p',
        output_path
    ])
    return output_path

def concat_videos(list_path, original_path, output_path):
    '''Creates a new video by concatenating the source chunks from the input
    list with the audio from the second input video.

    list_path(str) -- the path of the txt file with the list of video chunks
    origina-path(str) -- the path of the original video, to use to get the audio track
    output_path(str) -- the path of the output video to create'''

    call([
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0', # not really necessary
        '-i', list_path,
        '-i', original_path,
        '-c', '-copy',
        '-map', '0:0',
        '-map', '1:1',
        output_path
    ])
