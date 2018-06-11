from enum import Enum
from colorama import init, Fore

# initializes colorama
init(autoreset=True, convert=True)

def LOG(text):
    '''Displays a debug message to the screen, with the [DEBUG] prefix

    text(str) -- the message to display
    '''

    print('{}[DEBUG]'.format(Fore.LIGHTRED_EX), end=' ')
    print('{}{}'.format(Fore.LIGHTWHITE_EX, text), flush=True)

def INFO(text):
    '''Displays an info message, with the >> prefix

    text(str) -- the message to display
    '''

    print('{} >>'.format(Fore.LIGHTCYAN_EX), end=' ')
    print(text, flush=True)

class Progress(Enum):
    '''An enum that indicates a progress step when updating the UI
    '''

    START = 0
    STEP = 1
    END = 2

def PROGRESS(info):
    '''Displays a simple ASCII progress bar with an indefinite total progress

    info(Progress) -- the type of progress step to display
    '''

    if info == Progress.START:
        print('{}['.format(Fore.LIGHTYELLOW_EX), end='', flush=True)
    elif info == Progress.STEP:
        print('{}='.format(Fore.LIGHTYELLOW_EX), end='')
    else:
        print('{}]'.format(Fore.LIGHTYELLOW_EX), flush=True)
