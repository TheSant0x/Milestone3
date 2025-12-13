ERROR = 0
NORMAL = 1
WARNING = 2

verbosity = NORMAL
history = []
    
def setup(verbosity):
    verbosity = verbosity

def log(comment, level = NORMAL):
    history.append((level, comment))

    if(level <= verbosity):
        print(comment)

def getLast():
    '''
    Returns the last logged comment, without Level
    '''
    return history[-1][1]

def clear():
    history = []