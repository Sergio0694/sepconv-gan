import argparse
import os
import shutil
import tensorflow as tf
from subprocess import Popen, PIPE

OPS_GRADIENT_MAP = {
    'sepconv': True,
    'nearest_shader': False
}

def EXEC_AND_CHECK_ERRORS(args):
    split = args.split()
    reply, error = Popen(split, stdout=PIPE).communicate()
    if len(reply.decode('utf-8')) > 0 or error is not None:
        print('{}: {}, error: {}'.format(split[0], reply.decode('utf-8'), error))
        exit(-1)

def DELETE_IF_EXISTS(path):
    if os.path.isfile(path):
        os.remove(path)

def build(op):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    lib_training_path = os.path.join(file_dir, op, op)
    lib_inference_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(file_dir))), 'inference', 'src', 'ops', op)

    # cleanup
    DELETE_IF_EXISTS('{}.cu.o'.format(lib_training_path))
    DELETE_IF_EXISTS('{}_grad.cu.o'.format(lib_training_path))
    DELETE_IF_EXISTS('{}.so'.format(lib_training_path))
    DELETE_IF_EXISTS('{}.so'.format(lib_inference_path))

    # build
    TF_CFLAGS = " ".join(tf.sysconfig.get_compile_flags())
    TF_LFLAGS = " ".join(tf.sysconfig.get_link_flags())
    nvcc_args = 'nvcc -std=c++11 -c -o {}.cu.o {}.cu.cc {} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC'.format(lib_training_path, lib_training_path, TF_CFLAGS)
    EXEC_AND_CHECK_ERRORS(nvcc_args)

    if OPS_GRADIENT_MAP[op]:
        nvcc_args = 'nvcc -std=c++11 -c -o {}_grad.cu.o {}_grad.cu.cc {} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC'.format(lib_training_path, lib_training_path, TF_CFLAGS)
        EXEC_AND_CHECK_ERRORS(nvcc_args)
        gcc_args = 'g++ -std=c++11 -shared -o {}.so {}.cc {}_grad.cc {}.cu.o {}_grad.cu.o {} -fPIC -lcudart {}'.format(lib_training_path, lib_training_path, lib_training_path, lib_training_path, lib_training_path, TF_CFLAGS, TF_LFLAGS)
        EXEC_AND_CHECK_ERRORS(gcc_args)
    else:
        gcc_args = 'g++ -std=c++11 -shared -o {}.so {}.cc {}.cu.o {} -fPIC -lcudart {}'.format(lib_training_path, lib_training_path, lib_training_path, TF_CFLAGS, TF_LFLAGS)
        EXEC_AND_CHECK_ERRORS(gcc_args)

    # final cleanup
    DELETE_IF_EXISTS('{}.cu.o'.format(lib_training_path))
    DELETE_IF_EXISTS('{}_grad.cu.o'.format(lib_training_path))

    # copy the compiled op to the inference folder
    shutil.copy('{}.so'.format(lib_training_path), '{}.so'.format(lib_inference_path))

if __name__ == '__main__':

    # parse
    parser = argparse.ArgumentParser(description='Builds a TensorFlow GPU op from the input files.')
    parser.add_argument('-op', help='The custom op to build [sepconv|nearest_shader|all]', required=True)
    args = vars(parser.parse_args())

    # validate
    if args['op'] == 'all':
        for op in OPS_GRADIENT_MAP:
            build(op)
    elif args['op'] in [op for op in OPS_GRADIENT_MAP]:
        build(args['op'])
    else:
        raise ValueError('Invalid operation requested')
