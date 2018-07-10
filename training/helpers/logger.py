from colorama import init, Fore

# initializes colorama
init(autoreset=True)

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

def BAR(x, y, info=''):
    '''Displays a simple progress bar with

    x(int) -- the number of completed steps
    y(int) -- the total number of steps to perform
    int(str) -- additional info to print after the progress bar (optional)
    '''

    assert x >= 0 and x <= y

    if x < y:
        print('\r{}[{}{}]{}{}'.format(Fore.LIGHTYELLOW_EX, '=' * x, ' ' * (y - x), Fore.WHITE, info), end='', flush=True)
    else:
        print('', end='\r', flush=True) # reset the current line

def RESET_LINE(clean=False):
    '''Resets the current line by writing a carriage return character

    clean(bool) -- indicates whether or not to overwrite the current line to clean it up
    '''

    if clean:
        print('\r{}\r'.format(' ' * 100), end='', flush=True)
    else:
        print('\r', end='', flush=True)
