import os
import re
from shutil import copyfile, rmtree
import src.ffmpeg_helper as ffmpeg
from src.logger import LOG, INFO
from src.inference import process_frames, open_session

def frames_name_comparer(name):
    '''Compares two filenames and returns a tuple indicating
    their correct relative order to be used in the sort method.
    
    name(str) -- the current name to sort'''

    info = re.findall('([0-9]+)(_)?', name)[0]
    return (int(info[0]), info[1])

def format_duration(seconds):
    '''Returns a formatted string to indicate the duration of a video.'''

    return '{:02d}:{:02d}:{:02d}'.format(seconds // 3600, seconds // 60, seconds % 60)
    
def convert(args):
    '''Converts a video by doubling its framerate.

    params(dict<str, str>) -- the input parameters
    '''

    # Initial setup and info
    LOG('Processing {}'.format(args['source']))
    duration = ffmpeg.get_video_duration(args['source'])
    if duration < 10:
        raise ValueError('The video file is either empty or too short')
    INFO('Total duration: {}'.format(format_duration(duration)))

    # Loop until all video chunks have been created
    frames_path = os.path.join(args['working_dir'], 'frames')
    chunk_timestep, video_timestep, step_size = 1, 0, args['temp_buffer_lenght'] * 60
    chunks_paths = []
    with open_session(args['model_path'], frames_path) as session:
        while True:
            LOG('Converting {} to {}'.format(format_duration(video_timestep), format_duration(min(video_timestep + step_size, duration))))

            # Extract the frames from the n-th video chunk
            if os.path.isdir(frames_path):
                rmtree(frames_path)
            extract_ok = ffmpeg.extract_frames(
                args['source'], frames_path,
                [args['scale'], -1] if args['scale'] is not None else None,
                video_timestep, step_size, extension=args['frame_quality'], timeout=None)
            if not extract_ok: # this should never happen
                LOG('Failed to extract frames')
                exit(1)
            if not os.listdir(frames_path):
                break
            video_timestep += step_size

            # Inference pass on the n-th video chunk
            process_frames(frames_path, session)

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
                destination = '{}\\f{:03d}.{}'.format(frames_path, i, args['frame_quality'])
                os.rename(source, destination)

            # encode the interpolated video
            LOG('Encoding output video')
            chunk_path = os.path.join(args['working_dir'], '_{}.mp4'.format(chunk_timestep))
            ffmpeg.create_video('{}\\f%03d.{}'.format(frames_path, args['frame_quality']), chunk_path, args['encoder'], args['crf'], args['preset'])
            chunks_paths += [chunk_path]
            chunk_timestep += 1

    # prepare the list file
    LOG('Preparing final merge, {} chunk(s) available'.format(len(chunks_paths)))
    list_path = os.path.join(args['working_dir'], 'list.txt')
    with open(list_path, 'w', encoding='utf-8') as txt:
        for path in chunks_paths:
            print('file \'{}\''.format(path), file=txt)
    
    # create the final resampled video
    LOG('Creating output video')
    ffmpeg.concat_videos(list_path, args['source'], args['output'])
    rmtree(args['working_dir']) # cleanup
