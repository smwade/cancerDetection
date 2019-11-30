import os

CANCER_DATA_DIR = os.environ.get('CANCER_DATA_PATH')
CANCER_WORKDIR = os.environ.get('CANCER_WORDIR')

ABNORMAL_CELL_TYPES_SMEAR = [
    'severe_dysplastic',
    'carcinoma_in_situ',
    'moderate_dysplastic',
    'light_dysplastic'
]

NORMAL_CELL_TYPES_SMEAR = [
    'normal_columnar',
    'normal_intermediate',
    'normal_superficiel',
]
ABNORMAL_CELL_TYPES_SIP = [
    'koilocytotic',
    'metaplastic',
    'dyskeratotic'
]

NORMAL_CELL_TYPES_SIP = [
    'superficial-intermediate',
    'parabasal'
]

ABNORMAL_CELL_TYPES = ABNORMAL_CELL_TYPES_SMEAR + ABNORMAL_CELL_TYPES_SIP
NORMAL_CELL_TYPES = NORMAL_CELL_TYPES_SMEAR + NORMAL_CELL_TYPES_SIP