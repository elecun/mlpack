'''
 @brief     User Classification with Linear SVM
 @author    Byunghun Hwang<bh.hwang@iae.re.kr>
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import gc
from skimage import io, color
import tensorflow as tf
from tensorflow.keras.layers import MaxPool1D, GlobalMaxPooling1D
from sklearn import svm
from sklearn.datasets._samples_generator import make_blobs
from tensorflow.keras.layers import MaxPool1D, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
import random
from sklearn import metrics
from sklearn.metrics import precision_score


'''
Presets & Hyper-parameters
'''
CONFIGURATION_FILE_PATH = "./data/train/data_config.csv"
DATASET_PATH = "./data/train/"
pd.set_option('display.width', 200) # for display width
DYNAMIC_SCALEUP = False
INTERPOLATION_METHOD = "catrom"
CASE_PATH = "./catrom_static"
SAVE_FEATURE_IMAGE = True
FEATURE_LENGTH = 30 # n-dimensional data feature only use
NUMBER_OF_SAMPLES = 299 # number of augmented data
mu, sigma = 0, 1 # normal distribution random parameter for data augmentation
FEATURE_MAX_LENGTH = 115 # Maximum feature length
NUMBER_OF_RANDOM_SELECTION = 5
MAX_TRAIN_ITERATION = -1 # infinity
SVM_KERNEL_METHOD = 'linear'
NUMBER_OF_TESTING = 10
IMAGE_HEIGHT = 369


'''
1. Load raw data from configuration file
'''
data_config = pd.read_csv(CONFIGURATION_FILE_PATH, header=0, index_col=0)
print("Read configuration file shape : ", data_config.shape)

fsr_dataframe = {}
seat_dataframe = {}
for idx in data_config.index:
    fsr_filepath = DATASET_PATH+data_config.loc[idx, "fsr_matrix_1d_datafile"] # set FSR matrix data filepath
    seat_filepath = DATASET_PATH+data_config.loc[idx, "seat_datafile"] # set Seat data filepath
    print(idx, ") read data files : ", fsr_filepath, ",", seat_filepath)

    fsr_dataframe[idx] = pd.read_csv(fsr_filepath, header=0, index_col=False).iloc[:,0:162] # read FSR matrix data file
    seat_dataframe[idx] = pd.read_csv(seat_filepath, header=0, index_col=False) # read Seat data file

    del seat_dataframe[idx]['Measurement time'] # remove unnecessary column
    del fsr_dataframe[idx]['Measurement Time (sec)'] # remove unnecessary column


'''
2. extract segment from raw data
'''
fsr_dataframe_standard_segment = {}
fsr_dataframe_relax_segment = {}
seat_loadcell_dataframe_standard_segment = {}
seat_loadcell_dataframe_relax_segment = {}

for idx in data_config.index:
    mtime = data_config.loc[idx, ['standard_s_mtime', "standard_e_mtime", "relax_s_mtime", "relax_e_mtime"]]

    # seat loadcell segmentation
    seat_loadcell_dataframe_standard_segment[idx] = seat_dataframe[idx][(seat_dataframe[idx]['mtime']>=mtime.standard_s_mtime) & (seat_dataframe[idx]['mtime']<=mtime.standard_e_mtime)]
    seat_loadcell_dataframe_relax_segment[idx] = seat_dataframe[idx][(seat_dataframe[idx]['mtime']>=mtime.relax_s_mtime) & (seat_dataframe[idx]['mtime']<=mtime.relax_e_mtime)]

    # fsr matrix segmentation
    fsr_dataframe_standard_segment[idx] = fsr_dataframe[idx][(fsr_dataframe[idx]['mtime']>=mtime.standard_s_mtime) & (fsr_dataframe[idx]['mtime']<=mtime.standard_e_mtime)]
    fsr_dataframe_relax_segment[idx] = fsr_dataframe[idx][(fsr_dataframe[idx]['mtime']>=mtime.relax_s_mtime) & (fsr_dataframe[idx]['mtime']<=mtime.relax_e_mtime)]

    print("FSR Segments@Standard size : ", len(fsr_dataframe_standard_segment[idx]), ", FSR Segments@Relax size : ", len(fsr_dataframe_relax_segment[idx]))
    print("Seat Segments@Standard size : ", len(seat_loadcell_dataframe_standard_segment[idx]), ", Seat Segments@Relax size : ", len(seat_loadcell_dataframe_relax_segment[idx]))


'''
3. data crop & save data to 2d data
'''
crop_standard_interpolated_path = {}

try:
    os.mkdir(CASE_PATH) # create diretory
except FileExistsError:
    pass

for idx in data_config.index:
    fsr_standard_segment_1d = fsr_dataframe_standard_segment[idx].iloc[:,1:161]
    fsr_standard_segment_2d = fsr_standard_segment_1d.values.reshape(-1, 16, 10) # reshape

    try:
        os.mkdir("{}/{}".format(CASE_PATH, idx)) # create diretory for each id
    except FileExistsError:
        pass


    standard_fsr_crop_file_list = []

    for ridx in range(fsr_standard_segment_2d.shape[0]):
        result_image_filepath = "{}/{}/standard_{}.jpg".format(CASE_PATH, idx, ridx)
        result_crop_image_filepath = "{}/{}/standard_crop_{}.jpg".format(CASE_PATH, idx, ridx)

        # data interpolation
        if os.path.isfile(result_image_filepath)==False:
            fig = plt.figure()
            plt.axis('off')
            if DYNAMIC_SCALEUP==True:
                plt.imshow(fsr_standard_segment_2d[ridx], interpolation=INTERPOLATION_METHOD, cmap='Greys_r') # dynamic
            else:
                plt.imshow(fsr_standard_segment_2d[ridx], interpolation=INTERPOLATION_METHOD, vmin=0, vmax=255, cmap='Greys_r') # statuc
            plt.savefig(result_image_filepath, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            pass

        # crop active region
        if os.path.isfile(result_crop_image_filepath)==False:
            image = io.imread(result_image_filepath)
            grayscale = color.rgb2gray(image)
            crop = grayscale[0:grayscale.shape[0],int(grayscale.shape[1]/2):grayscale.shape[1]]
            io.imsave(result_crop_image_filepath, crop)
            print("saved output crop images for id {}, {}".format(idx, ridx))
        else:
            pass

        standard_fsr_crop_file_list.append(result_crop_image_filepath)
        
    crop_standard_interpolated_path[idx] = pd.DataFrame(standard_fsr_crop_file_list, columns=['path'])
    print(crop_standard_interpolated_path[idx])


    
'''
4. Feature extraction (1D max pooling)
'''
try:
    os.mkdir("{}/feature".format(CASE_PATH)) # create diretory
except FileExistsError:
    pass

featureset_container = {}
for idx in data_config.index:
    feature_set = np.array([], dtype=np.int64).reshape(0, IMAGE_HEIGHT)#empty(217)

    for f in crop_standard_interpolated_path[idx]["path"]:
        image = io.imread(f)
        grayscale = color.rgb2gray(image) # (217, 68)
        tensor = tf.reshape(grayscale, [grayscale.shape[0], grayscale.shape[1], 1]) # (217, 68, 1)
        feature = tf.keras.layers.GlobalMaxPooling1D()(tensor).numpy() #(217,1)
        feature_1d = feature.reshape(feature.shape[0]) #(217,)
        feature_set = np.vstack([feature_set, feature_1d])

    if SAVE_FEATURE_IMAGE == True:
        io.imsave("{}/feature/standard_feature1_{}.png".format(CASE_PATH, idx), feature_set.transpose())
    featureset_container[idx] = feature_set
    print("created featureset :", idx)


'''
5. more feature extraction & data augmentation
'''
feature_pt = random.choice(np.arange(FEATURE_MAX_LENGTH-FEATURE_LENGTH)) # random point to segment
print("feature segment point : {}".format(feature_pt))

augmented_data_dict = {}
for idx in data_config.index:
    feature_set = np.array([], dtype=np.int64).reshape(0, FEATURE_LENGTH)
    tensor = tf.reshape(featureset_container[idx], [featureset_container[idx].shape[0], featureset_container[idx].shape[1], 1]) # (1??, 217, 1)
    feature = tf.keras.layers.GlobalMaxPooling1D()(tensor).numpy() #(1??,1)
    feature_1d = feature.reshape(feature.shape[0]) #(1??,)
    feature_set = np.vstack([feature_set, feature_1d[feature_pt:FEATURE_LENGTH+feature_pt]])

    # data augmentation
    for aug in range(NUMBER_OF_SAMPLES):
        aug_1d = np.random.normal(mu, sigma, feature_1d.shape[0])
        feature_1d_aug = feature_1d + aug_1d
        #np.clip(feature_1d_aug, 0, None) # lower bound
        feature_set = np.vstack([feature_set, feature_1d_aug[feature_pt:FEATURE_LENGTH+feature_pt]])

    print("augmendted feature shape : {}".format(feature_set.shape))
    plt.figure()
    plt.plot(feature_set.transpose())
    plt.savefig("{}/feature/standard_feature_1d_{}.png".format(CASE_PATH, idx), bbox_inches='tight', pad_inches=0)

    if SAVE_FEATURE_IMAGE == True:
        io.imsave("{}/feature/standard_feature2_{}.png".format(CASE_PATH, idx), feature_set.transpose())
        

    augmented_data_dict[idx] = feature_set
    print(idx, ") generated data augmented :", augmented_data_dict[idx].shape)


'''
6. model training & testing
'''
for test_count in range(NUMBER_OF_TESTING):
    # random selection for testing
    shuffled_index = np.array(list(data_config.index))
    random.shuffle(shuffled_index)
    pclass = shuffled_index[0:NUMBER_OF_RANDOM_SELECTION] # first 5 index select from shuffled_index
    print("Selected Positive classes :", pclass)

    # data split
    Xcon_train = np.array([], dtype=np.int64).reshape(0, FEATURE_LENGTH)
    Xcon_test = np.array([], dtype=np.int64).reshape(0, FEATURE_LENGTH)
    ycon_train = np.array([], dtype=np.int64)
    ycon_test = np.array([], dtype=np.int64)

    for pc in data_config.index:
        X = augmented_data_dict[pc]
        if pc in pclass:
            y = np.full(X.shape[0], pc)
        else:
            y = np.full(X.shape[0], 0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)      
        Xcon_train = np.vstack([Xcon_train, X_train])
        ycon_train = np.hstack([ycon_train, y_train])
        Xcon_test = np.vstack([Xcon_test, X_test])
        ycon_test = np.hstack([ycon_test, y_test])

    print("model is training...")
    model = svm.SVC(kernel=SVM_KERNEL_METHOD, C=1, probability=True, max_iter=MAX_TRAIN_ITERATION, verbose=False)
    model.fit(Xcon_train,ycon_train)

    # testing results
    ypredict = model.predict(Xcon_test)
    print("Trial", test_count+1, ") Balanced Accuracy Score : ", metrics.balanced_accuracy_score(ycon_test, ypredict)*100)
    metrics.confusion_matrix(ycon_test, ypredict, normalize='all')
    metrics.plot_confusion_matrix(model, Xcon_test, ycon_test)
    plt.savefig("confusion_matrix_trial_{}.png".format(test_count+1))