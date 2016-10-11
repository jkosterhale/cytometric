#TYPE 

DROPBOX_LOCATION = '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/ActiveCode'
PREDICTIONS_LOCATION = '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/NotableLabsModels/automated-flow-alpha-master/predictions/'
MODELS_LOCATION = '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/NotableLabsModels/automated-flow-alpha-master/models/'

DATA_LOCATION = '/Users/jorie/Dropbox (Personal)/Insight_Personal/Analyses/Screens/screen_525/cell_plate_1/'

EVENT_IDENTIFYING_COLUMNS = ['screen_number', 'cell_plate_number', 'well_number','FSC-A', 'SSC-A', 'FSC-H', 'SSC-H']
RELEVANT_FEATURES = ['FSC-H', 'SSC-H', 'DAPI H', 'FSC-A', 'SSC-A', 'DAPI A']
TARGET = 'cell_type'
LABELS = {'blast', 'healthy'}