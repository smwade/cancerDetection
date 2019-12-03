import os
import shutil
from cancer.variables import CANCER_DATA_DIR


def delete_processed_data():
    out_dir = os.path.join(CANCER_DATA_DIR, 'SIPaKMeD', 'processed_data')
    shutil.rmtree(out_dir) 
    print('Deleted processed data.')

if __name__ == '__main__':
    delete_processed_data()