#DAD

DROPBOX_LOCATION = '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/ActiveCode'

DATA_LOCATION = '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/data/'

PREDICTIONS_LOCATION = '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/NotableLabsModels/automated-flow-alpha-master/predictions/'
MODELS_LOCATION = '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/NotableLabsModels/automated-flow-alpha-master/models/'



EVENT_IDENTIFYING_COLUMNS = ['screen_number', 'cell_plate_number', 'well_number',
                                 'FSC-A', 'SSC-A', 'FSC-H', 'SSC-H']
RELEVANT_FEATURES = ['FSC-H', 'SSC-H', 'DAPI H', 'FSC-A', 'SSC-A', 'DAPI A']
TARGET = 'cell_type'
LABELS = {'live', 'dead', 'debris'}
  