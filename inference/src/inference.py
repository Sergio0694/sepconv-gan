from multiprocessing import Process, Queue
from os import listdir
from time import time
import cv2
import tensorflow as tf
import src.dataset_loader as dataset
from src.logger import LOG, INFO, BAR, RESET_LINE

PROGRESS_BAR_LENGTH = 20

def save_frame(queue, extension):
    '''Saves a new frame in the background.'''

    while True:
        task = queue.get()
        if task is None:
            break

        # save frame in the correct format
        if extension == 'jpg':
            cv2.imwrite(task[0], task[1][0], [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        elif extension == 'png':
            cv2.imwrite(task[0], task[1][0], [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        else:
            cv2.imwrite(task[0], task[1][0]) # lossless .bmp image

def open_session(model_path, dataset_path):
    '''Loads a saved moden and opens a session for the inference pass.
    
    model_path(str) -- the path of the saved model
    dataset_path(str) -- the path to the dataset to process
    '''

    session = tf.Session()

    # restore the model from the .meta and check point files
    LOG('Restoring model')
    meta_file_path = [path for path in listdir(model_path) if path.endswith('.meta')][0]
    saver = tf.train.import_meta_graph('{}\\{}'.format(model_path, meta_file_path))
    saver.restore(session, tf.train.latest_checkpoint(model_path))

    # setup the input pipeline
    pipeline = dataset.setup_pipeline(dataset_path)
    inference_iterator = pipeline.make_initializable_iterator()     
    tf.add_to_collection('inference', inference_iterator)
    sample_tensor = tf.squeeze(inference_iterator.get_next(), axis=0)
    tf.add_to_collection('inference', sample_tensor)
    
    return session

def process_frames(working_path, session):
    
    # load the inference raw data
    LOG('Preparing frames')
    groups = dataset.load_samples(working_path, 1)
    previous_idx = len(groups[0]) // 2 - 1
    extension = groups[0][0][-4:] # same image format as the input
    INFO('{} frame pair(s) to process'.format(len(groups)))

    # setup the background worked
    frames_queue = Queue()
    worker = Process(target=save_frame, args=[frames_queue, extension])
    worker.start()

    # initialization
    LOG('Initialization')
    graph = tf.get_default_graph()
    pipeline_tensors = tf.get_collection('inference')
    pipeline_placeholder = graph.get_tensor_by_name('inference_groups:0')
    session.run(pipeline_tensors[0].initializer, feed_dict={pipeline_placeholder: groups})
    x = graph.get_tensor_by_name('x:0')
    yHat = graph.get_tensor_by_name('inference/uint8_img:0')

    # process the data
    LOG('Processing frames')
    BAR(0, PROGRESS_BAR_LENGTH)
    steps = 0
    start_seconds = time()
    for i, group in enumerate(groups):

        # load the current sample
        frames = session.run(pipeline_tensors[1])
        filename = group[previous_idx][:-4]

        # inference
        prediction = session.run(yHat, feed_dict={x: frames})
        frame_path = '{}\\{}_{}'.format(working_path, filename, extension)
        frames_queue.put((frame_path, prediction))

        # update the UI
        progress = (i * PROGRESS_BAR_LENGTH) // len(groups)
        if progress > steps:
            steps = progress
            BAR(steps, PROGRESS_BAR_LENGTH, ' | {0:.3f} fps'.format((i + 1) / (time() - start_seconds)))
    RESET_LINE(True)

    # wait for the background thread
    frames_queue.put(None)
    worker.join()
    frames_queue.close()
    LOG('Inference completed')
