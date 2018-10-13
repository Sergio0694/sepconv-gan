import argparse
import os
import re
import cv2
import matplotlib.pyplot as plt
from src.__MACRO__ import ERROR
from src.frame_inference import process_frames

def setup():
    '''Parses the command line arguments, validates them and generates the target frame.'''

    # get arguments
    parser = argparse.ArgumentParser(description='Generates and shows an intermediate frame for a given pair of consecutive frames.')
    parser.add_argument('-source', help='The path of the first frame. Its filename must end with a sequence number, so that the script will ' \
                        'be able to automatically retrieve the path for the following frame.', required=True)
    parser.add_argument('--model-path', default=None, help='The folder with the trained model to use.', required=True)
    parser.add_argument('--post-processing', default='default', help='The post-processing mode to apply to the generated frames [default|shader].', required=False)
    args = vars(parser.parse_args())

    # validate
    if not os.path.isfile(args['source']):
        ERROR('The input file does not exist')
    match = re.findall('([0-9]+)[.](jpg|png|bmp)$', args['source'])
    if not match:
        ERROR('The input path is not valid')
    index, extension = match[0]
    following = re.sub('[0-9]+[.](?:jpg|png|bmp)$', '{:03d}.{}'.format(int(index) + 1, extension), args['source'])
    print(following)
    if not os.path.isfile(following):
        ERROR('Couldn\'t find the following frame for the one in input')
    if args['model_path'] is None or not os.path.isdir(args['model_path']):
        ERROR('The input model directory does not exist.')
    if args['post_processing'] not in ['default', 'shader']:
        ERROR('Invalid post-processing mode selected')    
    
    # process and save the frame
    prediction = process_frames(args['model_path'], args['source'], following, args['post_processing'] == 'shader')
    save_path = os.path.join(os.path.dirname(args['source']), '{}_.{}'.format(index, extension))
    if extension == 'jpg':
        cv2.imwrite(save_path, prediction, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    elif extension == 'png':
        cv2.imwrite(save_path, prediction, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    else:
        cv2.imwrite(save_path, prediction) # any other extension

    # display the picture
    image = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    setup()    
