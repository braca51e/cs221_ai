import numpy as np
import termcolor

def colorRow(d):
    row = [d[s] for s in sorted(d)]
    toks = []
    for x in row:
        str_x = ('%.2f' % x).rjust(6)
        if x > 0:
            toks.append(termcolor.colored(str_x, 'green'))
        elif x < 0:
            toks.append(termcolor.colored(str_x, 'red'))
        else:
            toks.append(str_x)
    return ' '.join(toks)

def weightedSample(arr, probs):
    idx = np.random.choice(len(arr), p=np.array(probs))
    return arr[idx]

