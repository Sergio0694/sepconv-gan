import argparse
import os
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

def main():

    # parse
    parser = argparse.ArgumentParser(description='Builds a TensorFlow GPU op from the input files.')
    parser.add_argument('-op', help='The custom op to build [sepconv|nearest_shader]', required=True)
    args = vars(parser.parse_args())

    # validate
    if args['op'] not in [op for op in OPS_GRADIENT_MAP]:
        raise ValueError('Invalid operation requested')
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), args['op'], args['op'])

    # cleanup
    DELETE_IF_EXISTS('{}.cu.o'.format(lib_path))
    DELETE_IF_EXISTS('{}_grad.cu.o'.format(lib_path))
    DELETE_IF_EXISTS('{}.so'.format(lib_path))

    # build
    TF_CFLAGS = " ".join(tf.sysconfig.get_compile_flags())
    TF_LFLAGS = " ".join(tf.sysconfig.get_link_flags())
    nvcc_args = 'nvcc -std=c++11 -c -o {}.cu.o {}.cu.cc {} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC'.format(lib_path, lib_path, TF_CFLAGS)
    EXEC_AND_CHECK_ERRORS(nvcc_args)

    if OPS_GRADIENT_MAP[args['op']]:
        nvcc_args = 'nvcc -std=c++11 -c -o {}_grad.cu.o {}_grad.cu.cc {} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC'.format(lib_path, lib_path, TF_CFLAGS)
        EXEC_AND_CHECK_ERRORS(nvcc_args)
        gcc_args = 'g++ -std=c++11 -shared -o {}.so {}.cc {}_grad.cc {}.cu.o {}_grad.cu.o {} -fPIC -lcudart {}'.format(lib_path, lib_path, lib_path, lib_path, lib_path, TF_CFLAGS, TF_LFLAGS)
        EXEC_AND_CHECK_ERRORS(gcc_args)
    else:
        gcc_args = 'g++ -std=c++11 -shared -o {}.so {}.cc {}.cu.o {} -fPIC -lcudart {}'.format(lib_path, lib_path, lib_path, TF_CFLAGS, TF_LFLAGS)
        EXEC_AND_CHECK_ERRORS(gcc_args)

if __name__ == '__main__':
    main()
