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
    else:
        print('{}]'.format(Fore.LIGHTYELLOW_EX), flush=True)

def BAR(x, y):
    '''Displays a simple progress bar with

    x(int) -- the number of completed steps
    y(int) -- the total number of steps to perform
    '''

    assert x >= 0 and x <= y

    if x < y:
        print('\r{}[{}{}]'.format(Fore.LIGHTYELLOW_EX, '=' * x, ' ' * (y - x)), end='', flush=True)
    else:
        print('', end='\r', flush=True) # reset the current line

def RESET_LINE():
    '''Resets the current line by writing a carriage return character
    '''

    print('', end='\r', flush=True)
