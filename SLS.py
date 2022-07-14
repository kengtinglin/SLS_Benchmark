import numpy as np
import time

from utils.utils import cli
from data_gen.gen_bin import gen_table

def sls(object):
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
        rf = '>f4'
    elif arg.data_type == 'double':
        offset = 8
        rf = '>f8'

    total_calculation_time = 0
    table_read_time = 0
    for i in range(num_tables):
        np.random.seed(i)

        file_name = dir_path + '/' + 'EmbTable'+ str(i)
        indices = np.random.randint(
            0, ln_emb[i], np.sum(lengths)).astype(np.int64)
        data = []
        # print(f'### Read EmbTable{i} ###')
        with open(file_name, 'rb') as f:
            output = np.zeros((arg.mini_batch_size, arg.arch_sparse_feature_size),dtype=np.float32)
            if arg.lookup_mode == 'random':
                for j in range(arg.mini_batch_size):
                    for k in range(arg.num_indices_per_lookup):
                        tmp = offset * indices[j*arg.num_indices_per_lookup+k]*arg.arch_sparse_feature_size
                        read_beg = time.process_time()
                        f.seek(tmp)
                        output[j] += np.fromfile(f, rf, count=arg.arch_sparse_feature_size)
                        read_finish = time.process_time()
                        total_calculation_time += (read_finish-read_beg)
            elif arg.lookup_mode == 'special': 
                for j in range(arg.mini_batch_size):
                    indices[j*arg.num_indices_per_lookup:(j+1)*arg.num_indices_per_lookup] = np.sort(indices[j*arg.num_indices_per_lookup:(j+1)*arg.num_indices_per_lookup])
                    f.seek(0)
                    for k in range(arg.num_indices_per_lookup):
                        tmp = offset * indices[j*arg.num_indices_per_lookup+k]*arg.arch_sparse_feature_size - f.tell()
                        read_beg = time.process_time()
                        f.seek(tmp, 1)
                        output[j] += np.fromfile(f, rf, count=arg.arch_sparse_feature_size)
                        read_finish = time.process_time()
                        total_calculation_time += (read_finish-read_beg)
            elif arg.lookup_mode == 'all':
                read_beg = time.process_time()
                data = np.fromfile(f, rf).reshape((ln_emb[i], arg.arch_sparse_feature_size))
                read_finish = time.process_time()
                table_read_time += (read_finish-read_beg)     
                read_beg = time.process_time()
                for j in range(arg.mini_batch_size):
                    for k in range(arg.num_indices_per_lookup):
                        output[j] += data[indices[j*arg.num_indices_per_lookup+k]]
                read_finish = time.process_time()
                total_calculation_time += (read_finish-read_beg)
        # print(f'It spends {read_finish-read_beg}s')
        # print(f'### Finish Reading EmbTable{i} ###')
        # print('\n')
    if arg.lookup_mode == 'all':
        print(f'Read {num_tables} Table Spends {table_read_time}s.')
    print(f'Calculate {num_tables} Embedding Tables Spends {total_calculation_time}s.')
    
if __name__ == "__main__":
    args = cli()
    if args.gen_table == True:
        gen_table(args)
    sls(args)
