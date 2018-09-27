from preprocess import cal_unified_length, file_size_histogram
from preprocess import load_labels, process_part
import glob

"""
max length of bytes in trainset = 15417344
max size file is BrePaE2xAs9fJtqvN1Wp.bytes
"""


def preprocess_part_by_part():
    trainset_dir = 'trainSet'
    all_filenames = glob.glob(trainset_dir + '/*.bytes')
    part_size = 100
    num_parts = 4
    unified_length = cal_unified_length(all_filenames, part_size * num_parts)
    print('Unified byte length = %d (to pad)' % unified_length)
    label_mapping = load_labels('trainLabels.csv')
    for i in range(num_parts):
        process_part(i, part_size, unified_length,
                     all_filenames, label_mapping)


def plot_file_size_hist():
    trainset_dir = 'trainSet'
    all_filenames = glob.glob(trainset_dir + '/*.bytes')
    file_size_histogram(all_filenames)


plot_file_size_hist()
