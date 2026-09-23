import wfdb 
import os 

os.makedirs('data/raw', exist_ok=True) 
wfdb.dl_database('mitdb', dl_dir='data/raw') 
print('Download complete.')