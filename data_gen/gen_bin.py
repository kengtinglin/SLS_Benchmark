import struct
import numpy as np
import os

def gen_table(object):
    arg = object
    if arg.data_type == 'float32':
        dtype = np.float32
        wf = '>f'
    elif arg.data_type == 'double':
        dtype = np.double
        wf = '>d'

    dir_path = 'data/' + arg.model_name + '/' + arg.data_type
    print(dir_path)
    ln_emb = np.fromstring(arg.arch_embedding_size, dtype=int, sep="-")
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    if len(os.listdir(dir_path)) != 0:
        print('Data exists, exit the gen table process.')
        return
    num_tables = len(arg.arch_embedding_size.split("-"))
    print(f'Number of Embedding Tables: {num_tables}')


    for i in range(num_tables):
        file_name = dir_path + '/' + 'EmbTable'+ str(i)
        print(f'Generate {file_name}......')
        data = np.random.rand(ln_emb[i] * arg.arch_sparse_feature_size).astype(dtype)
        fout = open(file_name, 'ab')
        for j in range(ln_emb[i] * arg.arch_sparse_feature_size):
            fout.write(struct.pack(wf, data[j]))
        fout.close()
