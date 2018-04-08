from __future__ import division
import numpy as np
import glob
import pickle as pkl


def extract_byte_string(filename):
    f = open(filename, 'rb')
    contents = []
    num_lines = 0
    for line in f.readlines():
        num_lines += 1
        row = line.strip('\r\n').split(' ')
        cnt = 17 - len(line.split(' '))
        row += ["00" for _ in range(0, cnt)]
        contents += row[1:]  # discard line number in file

    f.close()
    print('Read %d lines from %s, padded to %d bytes' %
          (num_lines, filename, len(contents)))
    return contents


def tokenize(byte_string):
    char2int = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5,
                '6': 6, '7': 7, '8': 8, '9': 9,
                'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15}
    integer = []
    num_unknown = 0
    for b in byte_string:
        if b == '??':
            num_unknown += 1
            integer.append(256)
        else:
            integer.append(char2int[b[0]] * 16 + char2int[b[1]])

    print('Parsed %d tokens, in which %.2f%% are ??' %
          (len(integer), num_unknown / len(integer) * 100))
    return integer, num_unknown


def pad_zeros(content):
    num_paddings = unified_length - len(content)
    paddings = [0] * num_paddings
    print('Padded %.2f%% zeros to origin %d bytes' %
          (num_paddings / len(content) * 100, len(content)))
    content += paddings
    return num_paddings


def process_batch(batch_index):
    start_index = batch_index * batch_size
    end_index = min(len(all_filenames), (batch_index + 1) * batch_size)
    dataset = np.zeros((end_index - start_index, unified_length))
    metainfo = dict()
    for i in range(start_index, end_index):
        content = extract_byte_string(all_filenames[i])
        integer, num_unknown = tokenize(content)
        num_paddings = pad_zeros(integer)
        dataset[i - batch_index * batch_size, :] = integer
        metainfo[all_filenames[i]] = [num_unknown, num_paddings]

    print(dataset.shape)
    batch_result = {'dataset': dataset, 'metainfo': metainfo}
    pkl.dump(batch_result,
             open('trainset_batch_ind%d.pkl' % batch_index, 'wb'))


def cal_unified_length():
    result = 0
    for filename in all_filenames[:batch_size * num_batches]:
        content = extract_byte_string(filename)
        result = max(result, len(content))

    return result


"""
max length of bytes in trainset = 15417344
max size file is BrePaE2xAs9fJtqvN1Wp.bytes
"""
trainset_dir = 'trainSet'
all_filenames = glob.glob(trainset_dir + '/*.bytes')
batch_size = 10
num_batches = 3
unified_length = cal_unified_length()
print('Unified byte length = %d (to pad)' % unified_length)
for i in range(num_batches):
    process_batch(i)
