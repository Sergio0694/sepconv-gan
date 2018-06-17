import argparse
import os
import re
from shutil import copyfile, rmtree
from helpers.ffmpeg_helper import extract_all_frames, create_video
from helpers.logger import LOG, INFO, BAR, RESET_LINE
from inference import process_frames
    
def convert(params):
    '''Converts a video by doubling its framerate.

    params(dict<str, str>) -- the input parameters
    '''
    print(params)

    # extract the frames and execute the inference pass
    LOG('Processing {}'.format(params['source']))
    frames_path = extract_all_frames(params['source'], params['working_dir'], params['scale'], params['frame_quality'])
    process_frames(params['working_dir'], params['model_path'])

    # sort the frames by alternating the original and the interpolated
    LOG('Preparing generated frames')
    frames = os.listdir(params['working_dir'])

    def comparer(name):
        info = re.findall('([0-9]+)(_)?', name)[0]
        return (int(info[0]), info[1])
    frames.sort(key=comparer)

    # duplicate the last frame (no interpolation available)
    copy_filename = 'f{}_{}'.format(re.findall('([0-9]+)', frames[-1])[0], frames[-1][-4:])
    copyfile('{}\\{}'.format(params['working_dir'], frames[-1]), '{}\\{}'.format(params['working_dir'], copy_filename))
    frames += [copy_filename]
    INFO('{} total frame(s) to encode'.format(len(frames)))

    # rename the source frames to encode
    for i in range(len(frames), 0, -1):
        source = '{}\\{}'.format(params['working_dir'], frames[i - 1])
        destination = '{}\\f{:03d}.{}'.format(params['working_dir'], i, params['frame_quality'])
        os.rename(source, destination)

    # encode the interpolated video
    LOG('Encoding output video')
    create_video(frames_path, params['source'], params['output'], params['encoder'], params['crf'], params['preset'])
    LOG('{} created'.format(params['output']))

if __name__ == '__main__':

    # get arguments
    parser = argparse.ArgumentParser(description='Re-encode a source video with increased framerate.')
    parser.add_argument('-source', help='The video file to convert', required=True)
    parser.add_argument('--frame-quality', default='jpg', help='The format of intermediate frames [jpg|png|bmp].')
    parser.add_argument('-scale', default=-1, help='The optional scaling of the video (horizontal resolution).')
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
    if args['scale'] != -1:
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
    args['working_dir'] = os.path.join(args['working_dir'], '_frames')
    if os.path.isdir(args['working_dir']):
        rmtree(args['working_dir'])

    convert(args)
