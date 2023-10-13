#!/usr/bin/python

import sys
import struct
import numpy as np

def is_float(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

matrix = []
with open('dataset/glove.6B.300d.txt', 'r') as inf:
    with open('dataset/glove.6B.300d.dat', 'wb') as ouf:
        counter = 0
        for line in inf:
            # row2 = row = [float(x) for x in line.split()[1:]]
            # print(row2)
            row = [float(x) for x in line.split()[1:] if is_float(x)]
            if len(row) == 300:
                #assert len(row) == 300
                ouf.write(struct.pack('i', len(row)))
                ouf.write(struct.pack('%sf' % len(row), *row))
                counter += 1
                matrix.append(np.array(row, dtype=np.float32))
                if counter % 10000 == 0:
                    sys.stdout.write('%d points processed...\n' % counter)
            else:
                pass
np.save('dataset/glove.6B.300d', np.array(matrix))
