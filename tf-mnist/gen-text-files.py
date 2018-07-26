#!/usr/bin/env python
import pandas as pd
import os

def gen_text_data(rootdirs, file_path):
    if file_path.endswith('.txt'):
        text_file = open(file_path, 'w')
        dataset = trace_data(rootdirs)
        text_file.write('%s, %s\n' % (dataset['IMG_PATH'], dataset['LABEL']))
        text_file.close()
    elif file_path.endswith('.csv'):
        dataset = trace_data(rootdirs)
        data_frame = pd.DataFrame(dataset)
        data_frame.to_csv(file_path, sep=',', encoding='utf-8', index=False)

def trace_data(rootdirs):
    """Trace the path of image files in specified floder, and the label of that
       image.

    Arg:
        rootdirs (str): the root of specified floder path.

    Return:
        dataset (list with dicts): data tracing result.
    """
    dataset = []
    dirs_list = os.walk(rootdirs)
    for root, dirs, files in dirs_list:
        for f in files:
            path = os.path.join(root, f)
            label = os.path.relpath(root, rootdirs)
            if f.endswith(('.png','.jpg')):
                data = dict()
                data['IMG_PATH'] = path
                data['LABEL'] = label
                dataset.append(data)

if __name__ == '__main__':
    if not os.path.isdir('./text'):
        os.mkdir('./text')
    
    # Generate the text dataset.
    gen_text_data('images/mnist/training', './texts/mnist_train.txt')
    gen_text_data('images/mnist/testing', './texts/mnist_test.txt')

    # Generate the csv dataset.
    gen_text_data('images/mnist/training', './texts/mnist_train.csv')
    gen_text_data('images/mnist/testing', './texts/mnist_test.csv')
