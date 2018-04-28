from __future__ import division
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import os


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
    num_zeros = 0
    for b in byte_string:
        if b == '??':
            num_unknown += 1
            integer.append(256)
        else:
            num_zeros += (b == '00')
            integer.append(char2int[b[0]] * 16 + char2int[b[1]])

    print('Parsed %d tokens, in which %.2f%% are ??, %.2f%% are 00' %
          (len(integer), num_unknown / len(integer) * 100,
           num_zeros / len(integer) * 100))
    return integer, num_unknown


def pad_zeros(content, num_paddings):
    paddings = [0] * num_paddings
    print('Padded %.2f%% zeros[origin %d bytes] to unified vector length' %
          (num_paddings / len(content) * 100, len(content)))
    content += paddings
    return num_paddings


def process_part(part_index, part_size, unified_length,
                 all_filenames, label_mapping):
    start_index = part_index * part_size
    end_index = min(len(all_filenames), (part_index + 1) * part_size)
    dataset = np.zeros((end_index - start_index, unified_length))
    labels = np.zeros((end_index - start_index, 9))
    metainfo = dict()
    for i in range(start_index, end_index):
        content = extract_byte_string(all_filenames[i])
        integer, num_unknown = tokenize(content)
        num_paddings = pad_zeros(integer, unified_length - len(integer))

        byte_id = all_filenames[i].split('/')[1][:20]
        dataset[i - start_index, :] = integer
        labels[i - start_index, label_mapping[byte_id] - 1] = 1
        metainfo[byte_id] = [num_unknown, num_paddings]

    print(dataset.shape)
    partition = {'dataset': dataset, 'labels': labels, 'metainfo': metainfo}
    pkl.dump(partition,
             open('trainset_part_ind%d.pkl' % part_index, 'wb'))


def cal_unified_length(all_filenames, end_index):
    result = 0
    for filename in all_filenames[:end_index]:
        content = extract_byte_string(filename)
        result = max(result, len(content))

    return result


def load_labels(filename):
    label_mapping = dict()
    f = open(filename, 'rb')
    f.readline()  # skip column names
    for line in f.readlines():
        content = line.strip('\r\n').split(',')
        """strip "s around id"""
        label_mapping[content[0].strip('"')] = int(content[1])

    f.close()
    return label_mapping


def test_byte_process():
    content = extract_byte_string("trainSet/k9LJAopKrhzt8H3iIDuf.bytes")
    integer, num_unknown = tokenize(content)
    return pad_zeros(integer)
# unified_length = 320000
# test_byte_process()


def file_size_histogram(all_filenames):
    file_sizes = [os.stat(filename).st_size for filename in all_filenames]
    filtered = [x for x in file_sizes if x < 1e7]
    print('%.2f files is within 10M' %
          (len(filtered) / len(file_sizes)))
    print('Max size = %.2f Mb' % (max(file_sizes) / 1e6))
    n, bins, patches = plt.hist(filtered, 1000, histtype='step', alpha=0.7)
    plt.xlabel('File Sizes (Bytes)')
    plt.ylabel('#File Instances')
    plt.grid(True)
    plt.savefig('FileSizeHist.pdf', format='pdf', bbox_inches='tight')
