import argparse
import os
from shutil import rmtree
from dataset.dataset_builder import build_dataset

IMAGE_DIFF_MAX_THRESHOLD = 55
IMAGE_DIFF_MIN_THRESHOLD = 18
IMAGE_MIN_VARIANCE_THRESHOLD = 8
MAX_SUBSEQUENCE_LENGTH = 3

def setup():

    # get arguments
    parser = argparse.ArgumentParser(description='Builds a dataset in the specified location with the given parameters.')
    parser.add_argument('-source', nargs='+', help='The source folder that contains the video files. Multiple folders can be used, if needed.', required=True)
    parser.add_argument('--split-duration', help='The duration, in seconds, of each split video clip', required=True)
    parser.add_argument('-splits', help='The number of video clips to extract from each source video.', required=True)
    parser.add_argument('-resolution', default=None, help='The optional scaling of the video (horizontal resolution).')
    parser.add_argument('--frame-quality', default='jpg', help='The format of intermediate frames [jpg|png|bmp].')
    parser.add_argument('--min-variance', default=IMAGE_MIN_VARIANCE_THRESHOLD, help='The minimum variance value for a valid video frame')
    parser.add_argument('--min-diff-threshold', default=IMAGE_DIFF_MIN_THRESHOLD, help='The minimum difference between consecutive video frames')
    parser.add_argument('--max-diff-threshold', default=IMAGE_DIFF_MAX_THRESHOLD, help='The maximum difference between consecutive video frames')
    parser.add_argument('--max-subsequence-length', default=MAX_SUBSEQUENCE_LENGTH, help='The maximum length of a series of consecutive frames.')
    parser.add_argument('-output', help='The output path for the created dataset', required=True)
    args = vars(parser.parse_args())

    def ensure_valid_int(key, check, error):
        if type(args[key]) is int:
            return # default value
        if not args[key].isdigit() or not check(int(args[key])):
            raise ValueError(error)
        else:
            args[key] = int(args[key])

    # validate
    if not all((os.path.isdir(path) for path in args['source'])):
        raise ValueError('One of the input folders does not exist')
    ensure_valid_int('split_duration', lambda x: x >= 1, 'The duration of each split must be at least equal to 1')
    ensure_valid_int('splits', lambda x: x >= 1, 'The number of splits must be at least equal to 1')
    ensure_valid_int('resolution', lambda x: x >= 240, 'The resolution must be at least equal to 240')
    if args['frame_quality'] not in ['jpg', 'png', 'bmp']:
        raise ValueError('The frame quality must be either jpb, png or bmp.')
    args['frame_quality'] = '.{}'.format(args['frame_quality'])
    ensure_valid_int('min_variance', lambda x: x >= 0, 'The min variance must be a positive number')
    ensure_valid_int('min_diff_threshold', lambda x: x >= 0, 'The minimum difference threshold must be a positive number')
    ensure_valid_int('max_diff_threshold', lambda x: x >= args['min_diff_threshold'] + 100, 'The maximum difference threshold must be greater than the minimum threshold')
    ensure_valid_int('max_subsequence_length', lambda x: x >= 3, 'The maximum subsequence must be at least as wide as an input window (3)')
    if not os.path.isdir(args['output']):
        raise ValueError('The output folder does not exist')

    # cleanup
    for subdir in os.listdir(args['output']):
        full_path = os.path.join(args['output'], subdir)
        if os.path.isdir(full_path):
            rmtree(full_path)

    # execute
    build_dataset(
        args['source'], args['output'],
        args['split_duration'], args['splits'], args['resolution'], args['frame_quality'],
        args['min_variance'], args['min_diff_threshold'], args['max_diff_threshold'], args['max_subsequence_length'])

if __name__ == '__main__':
    setup()
