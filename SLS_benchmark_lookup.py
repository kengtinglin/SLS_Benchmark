import numpy as np
import time

from utils.utils import cli
from data_gen.gen_bin import gen_table

def sls_lookup(object):
    arg = object

    if arg.num_indices_per_lookup_fixed == True:
        lengths = arg.num_indices_per_lookup * np.ones(arg.mini_batch_size).astype(np.int32)
    else:
        lengths = np.random.randint(
            int(arg.num_indices_per_lookup * 0.75),
            int(arg.num_indices_per_lookup * 1.25),
            batch_size).astype(np.int32)

    dir_path = 'data/' + arg.model_name + '/' + arg.data_type
    ln_emb = np.fromstring(arg.arch_embedding_size, dtype=int, sep="-")
    num_tables = len(arg.arch_embedding_size.split("-"))

    if arg.data_type == 'float32':
        offset = 4
    elif arg.data_type == 'double':
        offset = 8

    total_read_time = 0
    for i in range(num_tables):
        file_name = dir_path + '/' + 'EmbTable'+ str(i)
        indices = np.random.randint(
            0, ln_emb[i], np.sum(lengths)).astype(np.int64)
        print(f'### Read EmbTable{i} ###')
        with open(file_name, 'rb') as f:
            f.seek(0)
            if arg.lookup_mode == 'random':
                read_beg = time.process_time()
                for j in range(arg.mini_batch_size):
                    for k in range(arg.num_indices_per_lookup):
                        f.seek(offset * indices[j*arg.num_indices_per_lookup+k])
                        f.read(offset * arg.arch_sparse_feature_size)
                read_finish = time.process_time()
            elif arg.lookup_mode == 'seq': 
                read_beg = time.process_time()
                for j in range(arg.mini_batch_size):
                    for k in range(arg.num_indices_per_lookup):
                        f.seek(0,1)
                        f.read(offset* arg.arch_sparse_feature_size)
                read_finish = time.process_time()
        print(f'read spends {read_finish-read_beg}s')
        total_read_time += (read_finish-read_beg)
        print(f'### Finish Reading EmbTable{i} ###')
        print('\n')
    print(f'Read {num_tables} Embedding Tables Spends {total_read_time}s.')
    
if __name__ == "__main__":
    args = cli()
    if args.gen_table == True:
        gen_table(args)
    sls_lookup(args)
