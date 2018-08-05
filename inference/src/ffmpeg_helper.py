import os
from pathlib import Path
from subprocess import call, Popen, PIPE, STDOUT, TimeoutExpired

def get_video_info(video_path):
    '''Returns a tuple with the framerate numerator and the video duration in seconds.

    video_path(str) -- the path of the video to analyze
    '''

    output = Popen(
        'ffprobe -v quiet -of csv=p=0 -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration "{}"'.format(video_path),
        shell=True,
        stdout=PIPE,
        stderr=STDOUT).communicate()
    try:
        info = output[0].decode('utf-8').strip().split(',') # bytes from PIPE > decode in utf-8
        return int(info[0]), int(info[1]), int(info[2].split('/')[0]), int(float(info[3]))  # [width, height, framerate, seconds]
    except ValueError:
        return None, None, None, None


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
        ['"{}"'.format(os.path.join(output_folder, '{}%05d.{}'.format(suffix, extension)))])

    output, err = Popen(' '.join(args), shell=True, stdout=PIPE, stderr=PIPE).communicate()
    return len(output) == 0 and len(err) == 0

def create_video(frames_path, output_path, in_fps, out_fps, encoder='h264', crf='23', preset='normal'):
    '''Creates an interpolated video from the input frames.

    frames_path(str) -- the formatted path of the folder with the source frames (returned by extract_all_frames)
    output_path(str) -- the path of the video file to create
    in_fps(str) -- the input framerate, in x/1001 format
    out_fps(str) -- the output framerate, in x/1001 format
    encoder(str) -- the encoder to use
    crf(str) -- the CRF value for the encoder
    preset(str) -- the encoder preset to use
    '''

    args = [
        'ffmpeg',
        '-y',
        '-loglevel', 'error',
        '-framerate', in_fps,
        '-start_number', '1',
        '-i', frames_path,
        '-c:v', 'libx{}'.format(encoder[1:]),
        '-crf', crf,
        '-preset', preset,
        '-r', out_fps,
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    Popen(' '.join(args), shell=True).communicate()

def concat_videos(list_path, original_path, output_path):
    '''Creates a new video by concatenating the source chunks from the input
    list with the audio from the second input video.

    list_path(str) -- the path of the txt file with the list of video chunks
    origina-path(str) -- the optional path of the original video, to use to get the audio track
    output_path(str) -- the path of the output video to create'''

    if original_path is None:
        args = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0', # not really necessary
            '-i', list_path,
            '-c', 'copy',
            output_path
        ]
    else:
        args = [
            'ffmpeg',
            '-f', 'concat',
            '-safe', '0', # not really necessary
            '-i', list_path,
            '-i', '"{}"'.format(original_path),
            '-c', 'copy',
            '-map', '0:v:0',
            '-map', '1:a:0',
            output_path
        ]
    
    Popen(' '.join(args), shell=True).communicate()
