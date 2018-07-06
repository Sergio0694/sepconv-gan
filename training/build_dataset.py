import argparse
import os
from __MACRO__ import IMAGE_MIN_VARIANCE_THRESHOLD, IMAGE_DIFF_MIN_THRESHOLD, IMAGE_DIFF_MAX_THRESHOLD
from dataset.dataset_builder import build_dataset

def setup():

    # get arguments
    parser = argparse.ArgumentParser(description='Builds a dataset in the specified location with the given parameters.')
    parser.add_argument('-source', nargs='+', help='The source folder that contains the video files. Multiple folders can be used, if needed.', required=True)
    parser.add_argument('--split-duration', help='The duration, in seconds, of each split video clip', required=True)
    parser.add_argument('-splits', help='The number of video clips to extract from each source video.', required=True)
    parser.add_argument('-resulution', default=None, help='The optional scaling of the video (horizontal resolution).')
    parser.add_argument('--frame-quality', default='jpg', help='The format of intermediate frames [jpg|png|bmp].')
    parser.add_argument('--min-variance', default=IMAGE_MIN_VARIANCE_THRESHOLD, help='The minimum variance value for a valid video frame')
    parser.add_argument('--min-diff-threshold', default=IMAGE_DIFF_MIN_THRESHOLD, help='The minimum difference between consecutive video frames')
    parser.add_argument('--max-diff-threshold', default=IMAGE_DIFF_MAX_THRESHOLD, help='The maximum difference between consecutive video frames')
    parser.add_argument('-output', help='The output path for the created dataset', required=True)
    args = vars(parser.parse_args())

    def ensure_valid_int(key, check, error):
        if not args[key].isdigit() or not check(int(args[key])):
            raise ValueError(error)
        else:
            args[key] = int(args[key])

    # validate
    if not all((os.path.isdir(path) for path in args['source'])):
        raise ValueError('One of the input folders does not exist')
    ensure_valid_int('split_duration', lambda x: x >= 1, 'The duration of each split must be at least equal to 1')
    ensure_valid_int('splits', lambda x: x >= 1, 'The number of splits must be at least equal to 1')
    ensure_valid_int('resulution', lambda x: x >= 240, 'The resulution must be at least equal to 240')
    if args['frame_quality'] not in ['jpg', 'png', 'bmp']:
        raise ValueError('The frame quality must be either jpb, png or bmp.')
    ensure_valid_int('min_variance', lambda x: x >= 0, 'The min variance must be a positive number')
    ensure_valid_int('min_diff_threshold', lambda x: x >= 0, 'The minimum difference threshold must be a positive number')
    ensure_valid_int('max_diff_threshold', lambda x: x >= args['min_diff_threshold'] + 100, 'The maximum difference threshold must be greater than the minimum threshold')
    if not os.path.isdir(args['output']):
        raise ValueError('The output folder does not exist')

    # execute
    build_dataset(
        args['source'], args['output'],
        args['split_duration'], args['splits'], args['resolution'], args['frame_quality'],
        args['min_variance'], args['min_diff_threshold'], args['max_diff_threshold'])

if __name__ == '__main__':
    setup()
