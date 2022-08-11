# SLS_Benchmark
This is a SLS benchmark which use two access methods, directly from storage and from DRAM, respectively.

## How to use
- Generate embedding table from config and execute
  ```sh
    python3 SLS.py --config_file models/rm1.json --lookup-mode all --gen-table
  ```
