import sys
import pdb

filename = str(sys.argv[1])
first_line = int(sys.argv[2])

train_token_order = 3
test_token_order = 5
# read_span = 30  # number of lines _between_ the two repeated target lines

f = open(filename, 'r')
# for i in range(first_line - 1): trashbox = f.readline()

try:
    while True:
        line = f.readline()
        tokens = line.split()
        if len(tokens) != 0 and tokens[0] == '***error':
            print tokens[train_token_order][:-1] + '\t',
            print tokens[test_token_order]
        elif line == '':
            break
        else:
            continue
except IndexError, e:
    print "Error tokenizing line: \n%s\n" % line, e

