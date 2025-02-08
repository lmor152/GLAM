import numpy as np
import random

# dictionary containing nearby keys for substitution typos (all are equally likely)
nearby_keys = {
    'a' : 'qwsz'
    ,'b' : 'nhgv '
    ,'c' : 'vfdx '
    ,'d' : 'fresxc'
    ,'e' : 'sdfr43ws'
    ,'f' : 'gtrdcv'
    ,'g' : 'hytfvb'
    ,'h' : 'juytgbn'
    ,'i' : 'ujklo98'
    ,'j' : 'mkiuyhn'
    ,'k' : 'jm,loij'
    ,'l' : 'k,.;pok'
    ,'m' : 'njk, '
    ,'n' : 'bhjm '
    ,'o' : 'plki90p'
    ,'p' : 'ol;[-0o'
    ,'q' : 'asw21'
    ,'r' : 'tfde45'
    ,'s' : 'dxzawe'
    ,'t' : 'ygfr56'
    ,'u' : 'ijhy78'
    ,'v' : 'cfgb '
    ,'w' : 'saq23e'
    ,'x' : 'zsdc'
    ,'y' : 'uhgt67'
    ,'z' : 'xsa'
    ,'0' : '-po9'
    ,'1' : '2q'
    ,'2' : '3wq1'
    ,'3' : '4ew2'
    ,'4' : '5re3'
    ,'5' : '6tr4'
    ,'6' : '7yt5'
    ,'7' : '8uy6'
    ,'8' : '9iu7'
    ,'9' : '0oi8'
}

def generate_typo(s, substitution_rate = 0.005, deletion_rate = 0.005, duplication_rate = 0.005, transpose_rate = 0.005):
    """
    Takes in a string and generates a new string with random reasonable typos added according to input probabilities
    
    Inputs:
        s:                 the input string
        substitution_rate: probability of substituting the character for another nearby on a keyboard [0 < x < 1]
        deletion_rate:     probability of deleting a character  [0 < x < 1]
        duplication_rate:  probability of duplicating a character [0 < x < 1]
        transpose_rate:    probability of switching two characters [0 < x < 1]
    Outputs:
        return:            new string with typos added
    """
    
    if len(s) == 0:
        return s

    s_new = []
    for i, char in enumerate(s.lower()):

        # Decide what to do
        r = np.random.uniform(size=(4,))
        do_swap = r[0] <= substitution_rate
        do_del  = r[1] <= deletion_rate
        do_dupe = r[2] <= duplication_rate
        do_T    = r[3] <= transpose_rate

        if do_swap and char in nearby_keys:
            # substitute current character for nearby key
            s_new.append(random.choice(nearby_keys[char]))

        elif do_del:
            # do not add current character
            continue

        elif do_dupe:
            # add current character twice
            s_new += [char]*2

        elif do_T and len(s_new) > 0:
            # switch current character with previous character (if not at start of string)
            s_new.append(s_new[-1])
            s_new[-2] = char

        else:
            # Keep the character
            s_new.append(char)

    return ''.join(s_new)
