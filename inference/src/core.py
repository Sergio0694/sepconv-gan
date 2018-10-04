import os
import re
from shutil import copyfile, rmtree
import src.ffmpeg_helper as ffmpeg
from src.__MACRO__ import LOG, INFO, ERROR
from src.video_inference import process_frames, open_session

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
    width, height, framerate, duration = ffmpeg.get_video_info(args['source'])
    if any((info is None for info in [width, height, framerate, duration])):
        ERROR('Error retrieving video info')
    if duration < 2:
        ERROR('The video file is either empty or too short')
    INFO('Total duration: {}'.format(format_duration(duration)))
    INFO('Framerate: {}/1001'.format(framerate))
    INFO('Resolution: {}x{}'.format(width, height))

    # Validate the target resolution
    if args['scale'] is not None:
        if (args['scale'] * height) % width != 0:
            ERROR('The scaled resolution would produce a fractional height')
        if (args['scale'] * height // width) % 4 != 0:
            ERROR('The scaled resolution must be a multiple of 4 to be encoded correctly')

    # calculate the framerate parameters to encode the video chunks
    if args['interpolation'] == 'double':
        in_fps, out_fps = '{}/1001'.format(framerate * 2), '{}/1001'.format(framerate * 2)
    else:
        in_fps, out_fps = '{}/1001'.format(framerate), '{}/1001'.format(framerate)

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
                video_timestep, step_size, extension=args['frame_quality'])

            # progress checks
            if not extract_ok or (video_timestep == 0 and not os.listdir(frames_path)):
                rmtree(args['working_dir'])
                ERROR('Failed to extract frames')
            video_timestep += step_size

            # Inference pass on the n-th video chunk
            if not os.listdir(frames_path):
                break
            process_frames(frames_path, session, args['post_processing'] == 'shader')

            # sort the frames by alternating the original and the interpolated
            LOG('Preparing generated frames')     
            frames = os.listdir(frames_path)               
            frames.sort(key=frames_name_comparer)

            # duplicate the last frame (no interpolation available)
            copy_filename = '{}_{}'.format(re.findall('([0-9]+)', frames[-1])[0], frames[-1][-4:])
            copyfile(os.path.join(frames_path, frames[-1]), os.path.join(frames_path, copy_filename))
            frames += [copy_filename]
            INFO('{} total frame(s) to encode'.format(len(frames)))

            # rename the source frames to encode
            for i in range(len(frames), 0, -1):
                source = os.path.join(frames_path, frames[i - 1])
                destination = os.path.join(frames_path, '{:05d}.{}'.format(i, args['frame_quality']))
                os.rename(source, destination)

            # encode the interpolated video
            LOG('Encoding video chunk #{}'.format(video_timestep // step_size))
            chunk_path = os.path.join(args['working_dir'], '_{}.mp4'.format(chunk_timestep))
            ffmpeg.create_video(os.path.join(frames_path, '%05d.{}'.format(args['frame_quality'])), chunk_path, in_fps, out_fps, args['encoder'], args['crf'], args['preset'])
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
    ffmpeg.concat_videos(list_path, args['source'] if args['interpolation'] == 'double' else None, args['output'])
    rmtree(args['working_dir']) # cleanup
    LOG('Video creation completed')
