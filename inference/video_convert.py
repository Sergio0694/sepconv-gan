import argparse
import os
from shutil import rmtree
from src.core import convert
from src.__MACRO__ import ERROR

def setup():
    '''Parses the command line arguments, validates them and starts the conversion process.'''

    # get arguments
    parser = argparse.ArgumentParser(description='Re-encode a source video with increased framerate.')
    parser.add_argument('-source', help='The video file to convert', required=True)
    parser.add_argument('--frame-quality', default='bmp', help='The format of intermediate frames [jpg|png|bmp].')
    parser.add_argument('-scale', default=None, help='The optional scaling of the video (horizontal resolution).')
    parser.add_argument('--model-path', default=None, help='The folder with the trained model to use.', required=True)    
    parser.add_argument('--working-dir', default=None, help='An optional path for the working dir to use to store temporary files.')
    parser.add_argument('--temp-buffer-lenght', default=1, help='The maximum duration of temporary video buffers to store on disk.')
    parser.add_argument('-encoder', default='h264', help='The video encoder [h264|h265].')
    parser.add_argument('-crf', default='23', help='The CRF quality for the encoded video [0-51].')
    parser.add_argument('-preset', default='medium', help='The encoding profile (see trac.ffmpeg.org/wiki/Encode/H.264).')
    parser.add_argument('-interpolation', default='double', help='The ratio between input and output framerate [double|half]. ' \
                        '"double" keeps the original video length and doubles the framerate, while "half" maintains the input ' \
                        'framerate and makes the output video twice as long as the original, effectively slowing it down. ' \
                        'Using this mode will remove the audio from the output video, as it wouldn\'t have the right duration.')
    parser.add_argument('--post-processing', default='default', help='The post-processing mode to apply to the generated frames [default|shader].')
    parser.add_argument('-output', help='The path of the output file to create', required=True)
    args = vars(parser.parse_args())

    # validate
    if not os.path.isfile(args['source']):
        ERROR('The input file does not exist')
    if args['frame_quality'] not in ['jpg', 'png', 'bmp']:
        ERROR('The frame quality must be either jpb, png or bmp.')
    if args['scale'] != None:
        if not args['scale'].isdigit() or int(args['scale']) < 240:
            ERROR('The scale must be at least equal to 240.')
        else:
            args['scale'] = int(args['scale'])
    if args['model_path'] is None or not os.path.isdir(args['model_path']):
        ERROR('The input model directory does not exist.')
    if args['working_dir'] is None:
        args['working_dir'] = os.path.dirname(args['output'])
    if args['temp_buffer_lenght'] != 1:
        if not args['temp_buffer_lenght'].isdigit():
            ERROR('Invalid temp buffer length')
        else:
            args['temp_buffer_lenght'] = int(args['temp_buffer_lenght'])
    if args['encoder'] not in ['h264', 'h265']:
        ERROR('The encoder must be either h264 or h265')
    if not args['crf'].isdigit() or not 0 <= int(args['crf']) <= 51:
        ERROR('The CRF value must be in the [0-51] range.')
    if args['preset'] not in ['ultrafast' 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']:
        ERROR('Invalid preset value, see trac.ffmpeg.org/wiki/Encode/H.264 for more info.')
    if args['interpolation'] not in ['double', 'half']:
        ERROR('Invalid interpolation option selected')
    if args['post_processing'] not in ['default', 'shader']:
        ERROR('Invalid post-processing mode selected')
    if not args['output'].endswith('.mp4'):
        ERROR('The output file must have the .mp4 extension')
    
    # delete leftovers and execute
    args['working_dir'] = os.path.join(args['working_dir'], '_temp')
    if os.path.isdir(args['working_dir']):
        rmtree(args['working_dir'])
    os.mkdir(args['working_dir'])
    convert(args)

if __name__ == '__main__':
    setup()    
