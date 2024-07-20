from pathlib import Path  # pathlib is seriously awesome!



_local_dir = True # in case you want to redirect the output somewhere else
# True False

if _local_dir:
    data_dir = Path(__file__).resolve().parents[1] / 'data'
    reportings_dir = Path(__file__).resolve().parents[1].joinpath('reports')
    log_dir = Path(__file__).resolve().parents[2].joinpath('log_files')
    
else:
    data_dir = Path(__file__).resolve().parents[4].joinpath('Projects','Local Code', 'data')
    raw_data_dir = Path(__file__).resolve().parents[4].joinpath('Projects', 'Local Code', 'data', 'raw')
    reportings_dir = Path(__file__).resolve().parents[4].joinpath('Projects', 'Local Code', 'reports')
    log_dir = Path(__file__).resolve().parents[4].joinpath('Projects', 'Local Code', 'data', 'log_files')


assert data_dir.is_dir()
assert reportings_dir.is_dir()
assert log_dir.is_dir()

#print(data_dir.is_dir())
#print(data_dir)


#(raw_dir / 'realpython.txt').is_file()


#data_dir = Path('/path/to/some/logical/parent/dir')
#data_path = data_dir / 'my_file.csv'  # use feather files if possible!!!

#customer_db_url = 'sql:///customer/db/url'
#purchases_db_url = 'sql:///purchases/db/url'
