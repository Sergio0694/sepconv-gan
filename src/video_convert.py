import argparse
import os
import re
from shutil import copyfile, rmtree
import helpers.ffmpeg_helper as ffmpeg
from helpers.logger import LOG, INFO, BAR, RESET_LINE
from inference import process_frames

def frames_name_comparer(name):
    info = re.findall('([0-9]+)(_)?', name)[0]
    return (int(info[0]), info[1])

def format_duration(seconds):
    return '{:02d}:{:02d}:{:02d}'.format(seconds // 3600, seconds // 60, seconds % 60)
    
def convert(params):
    '''Converts a video by doubling its framerate.

    params(dict<str, str>) -- the input parameters
    '''

    # Initial setup and info
    LOG('Processing {}'.format(params['source']))
    duration = ffmpeg.get_video_duration(args['source'])
    if duration == 0:
        raise ValueError('Invalid video file')
    INFO('Total duration: {}'.format(format_duration(duration)))

    # Loop until all video chunks have been created
    frames_path = os.path.join(args['working_dir'], 'frames')
    chunk_timestep, video_timestep, step_size = 1, 0, args['temp_buffer_lenght'] * 20
    chunks_paths = []
    while True:
        LOG('Converting {} to {}'.format(format_duration(video_timestep), format_duration(min(video_timestep + step_size, duration))))

        # Extract the frames from the n-th video chunk
        if os.path.isdir(frames_path):
            rmtree(frames_path)
        extract_ok = ffmpeg.extract_frames(
            params['source'], frames_path,
            [args['scale'], -1] if args['scale'] is not None else None,
            video_timestep, step_size, extension=args['frame_quality'])
        if not extract_ok: # this should never happen
            exit(1)
        if not os.listdir(frames_path):
            break
        video_timestep += step_size

        # Inference pass on the n-th video chunk
        process_frames(frames_path, params['model_path'] if not chunks_paths else None) # avoid reloading the model

        # sort the frames by alternating the original and the interpolated
        LOG('Preparing generated frames')
        frames = os.listdir(frames_path)        
        frames.sort(key=frames_name_comparer)

        # duplicate the last frame (no interpolation available)
        copy_filename = 'f{}_{}'.format(re.findall('([0-9]+)', frames[-1])[0], frames[-1][-4:])
        copyfile('{}\\{}'.format(frames_path, frames[-1]), '{}\\{}'.format(frames_path, copy_filename))
        frames += [copy_filename]
        INFO('{} total frame(s) to encode'.format(len(frames)))

        # rename the source frames to encode
        for i in range(len(frames), 0, -1):
            source = '{}\\{}'.format(frames_path, frames[i - 1])
            destination = '{}\\f{:03d}.{}'.format(frames_path, i, params['frame_quality'])
            os.rename(source, destination)

        # encode the interpolated video
        LOG('Encoding output video')
        chunk_path = os.path.join(args['working_dir'], '_{}.mp4'.format(chunk_timestep += 1))
        ffmpeg.create_video('{}\\f%03d.{}'.format(frames_path, params['frame_quality']), chunk_path, params['encoder'], params['crf'], params['preset'])
        chunks_paths += [chunk_path]
        chunk_timestep += 1

    # prepare the list file
    list_path = os.path.join(args['working_dir'], 'list.txt')
    with open(list_path, 'w', encoding='utf-8') as txt:
        for path in chunks_paths:
            print('file \'{}\''.format(path), file=txt)
    
    # create the final resampled video
    ffmpeg.concat_videos(list_path, args['source'], args['output'])

if __name__ == '__main__':

    # get arguments
    parser = argparse.ArgumentParser(description='Re-encode a source video with increased framerate.')
    parser.add_argument('-source', help='The video file to convert', required=True)
    parser.add_argument('--frame-quality', default='jpg', help='The format of intermediate frames [jpg|png|bmp].')
    parser.add_argument('-scale', default=None, help='The optional scaling of the video (horizontal resolution).')
    parser.add_argument('--model-path', default=r'\model', help='The folder with the trained model to use.', required=True)    
    parser.add_argument('--working-dir', default=None, help='An optional path for the working dir to use to store temporary files.')
    parser.add_argument('--temp-buffer-lenght', default=1, help='The maximum duration of temporary video buffers to store on disk.')
    parser.add_argument('-encoder', default='h264', help='The video encoder [h264|h265].')
    parser.add_argument('-crf', default='23', help='The CRF quality for the encoded video [0-51].')
    parser.add_argument('-preset', default='medium', help='The encoding profile (see trac.ffmpeg.org/wiki/Encode/H.264).')    
    parser.add_argument('-output', help='The path of the output file to create', required=True)
    args = vars(parser.parse_args())

    # validate
    if not os.path.isfile(args['source']):
        raise ValueError('The input file does not exist')
    if args['frame_quality'] not in ['jpg', 'png', 'bmp']:
        raise ValueError('The frame quality must be either jpb, png or bmp.')
    if args['scale'] != None:
        if not args['scale'].isdigit() or int(args['scale']) < 240:
            raise ValueError('The scale must be at least equal to 240.')
        else:
            args['scale'] = int(args['scale'])
    if not os.path.isdir(args['model_path']):
        raise ValueError('The input model directory does not exist.')
    if args['working_dir'] is None:
        args['working_dir'] = os.path.dirname(args['output'])
    if args['temp_buffer_lenght'] != 1:
        if not args['temp_buffer_lenght'].isdigit():
            raise ValueError('Invalid temp buffer length')
        else:
            args['temp_buffer_lenght'] = int(args['temp_buffer_lenght'])
    if args['encoder'] not in ['h264', 'h265']:
        raise ValueError('The encoder must be either h264 or h265')
    if not args['crf'].isdigit() or not 0 <= int(args['crf']) <= 51:
        raise ValueError('The CRF value must be in the [0-51] range.')
    if args['preset'] not in ['ultrafast' 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']:
        raise ValueError('Invalid preset value, see trac.ffmpeg.org/wiki/Encode/H.264 for more info.')
    if not args['output'].endswith('.mp4'):
        raise ValueError('The output file must have the .mp4 extension')
    
    # delete leftovers and execute
    args['working_dir'] = os.path.join(args['working_dir'], '_temp')
    if os.path.isdir(args['working_dir']):
        rmtree(args['working_dir'])
    os.mkdir(args['working_dir'])
    convert(args)
