from __future__ import absolute_import, division, print_function
import os, sys, pdb, pickle
from multiprocessing import Pool
import numpy as np
import samplerate
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt
import csv
import pdb
np.seterr(all='raise')

from sets import Set

# INPUT DIRECTORIES
train_data_dir = '../ESC-50/'

# OUTPUT INFORMATION
data_folder = '../ESC_outfiles/'
data_version = 'processed-ESC50_tdfilt_tdmix_albertspect'

#### Standard Spectrogram #####
fnames = [
    '101 - Dog/1-100032-A-0.csv',
    '201 - Rain/1-17367-A-0.csv',
    '101 - Dog/1-110389-A-0.csv',
    '201 - Rain/1-21189-A-0.csv',
    '101 - Dog/1-30226-A-0.csv',
    '201 - Rain/1-26222-A-0.csv',
]

#plt.figure(figsize=(20, 18))
#for i in range(6):
#    rate, data = wav.read(train_data_dir + fnames[i])
#    plt.subplot(3,2,i+1)
#    spec = plt.specgram(data, Fs=rate, NFFT=1024, noverlap=1024-160, mode='psd')
#    plt.yscale('log')
#    plt.ylim([10,8000])
#    plt.title(fnames[i])
#plt.show()

########## Paritition Dataset ############


######### Import Data From CSV ###########
def conv_csv_to_img(filename):
    spectrogram = []
    if '\0' in open(filename).read():
        print("wtf " + filename)
    with open(filename, 'rb') as csv_file:
        csv_obj = csv.reader(csv_file)
        for row in csv_obj:
            spectrogram.append(row)
    spectrogram = (np.array(spectrogram)).astype('f4')
    return spectrogram.astype('f4')

plt.figure(figsize=(20, 12))
for i in range(6):
    plt.subplot(3,2,i+1)
    plt.imshow(conv_csv_to_img(train_data_dir + fnames[i]), interpolation='nearest', origin='lower')
    plt.title('Mel ' + fnames[i])
plt.show()

########## Main Code for Generating Training Dataset ##########
np.random.seed(0)

size_multiplier = 5
#classes = [('%s'%fname) for fname in os.listdir(train_data_dir) if fname.split('.')[0] == fname.split('.')[-1]]
classes = ['101 - Dog', '102 - Rooster', '103 - Pig', '104 - Cow', '105 - Frog', '106 - Cat', '107 - Hen', '108 - Insects', '109 - Sheep', '110 - Crow', '201 - Rain', '202 - Sea waves', '203 - Crackling fire', '204 - Crickets', '205 - Chirping birds', '206 - Water drops', '207 - Wind', '208 - Pouring water', '209 - Toilet flush', '210 - Thunderstorm', '301 - Crying baby', '302 - Sneezing', '303 - Clapping', '304 - Breathing', '305 - Coughing', '306 - Footsteps', '307 - Laughing', '308 - Brushing teeth', '309 - Snoring', '310 - Drinking - sipping', '401 - Door knock', '402 - Mouse click', '403 - Keyboard typing', '404 - Door - wood creaks', '405 - Can opening', '406 - Washing machine', '407 - Vacuum cleaner', '408 - Clock alarm', '409 - Clock tick', '410 - Glass breaking', '501 - Helicopter', '502 - Chainsaw', '503 - Siren', '504 - Car horn', '505 - Engine', '506 - Train', '507 - Church bells', '508 - Airplane', '509 - Fireworks', '510 - Hand saw']

def get_train(x):
    i, fname = x
    print('\rOn %d'%(i+1), end='')
    sys.stdout.flush()
    xs = []
    xs.append(conv_csv_to_img(fname))
    for j in range(size_multiplier - 1):
        xs.append(conv_csv_to_img(fname))
    return xs

if __name__ == "__main__":
    pool = Pool(8)
    Xtr = []
    Xva = []
    Xte = []
    ytr = []
    yva = []
    yte = []
    for i,c in enumerate(classes):
        print('Processing "%s"'%c)
        all_fnames = [('%s'%fname) for fname in os.listdir(train_data_dir + c) if fname.split('.')[-1] == 'csv']
        fnames_tr = []
        fnames_va = []
        fnames_te = []
        for fname in all_fnames:
            ffold = fname.split('-')[0]
            fhash = fname.split('-')[1]
            if ffold == '1': fnames_va.append(fname)
            elif ffold == '2': fnames_te.append(fname)
            #if fhash in va[c]: fnames_va.append(fname)
            #elif fhash in te[c]: fnames_te.append(fname)
            else: fnames_tr.append(fname)

        print('On Training (total %d)'%(size_multiplier * len(fnames_tr)))
        sys.stdout.flush()
        xsa = pool.map(get_train, enumerate(map(lambda fname: train_data_dir + '%s/%s'%(c, fname), fnames_tr)))
        xsf = [x for xs in xsa for x in xs]
        Xtr += xsf
        ytr += [i] * len(xsf)
        print()
        for ix, fname in enumerate(fnames_va):
            print('\rOn Validation %d/%d'%(ix+1,len(fnames_va)), end='')
            Xva.append(conv_csv_to_img(train_data_dir + '%s/%s'%(c, fname)))
            yva.append(i)
        print()
        for ix, fname in enumerate(fnames_te):
            print('\rOn Testing %d/%d'%(ix+1,len(fnames_te)), end='')
            Xte.append(conv_csv_to_img(train_data_dir + '%s/%s'%(c, fname)))
            yte.append(i)
        print()

    # Final processing, shuffling, saving.
    shuffle = np.arange(len(Xtr))
    np.random.shuffle(shuffle)
    Xtr = (np.array(Xtr).astype('f4'))[shuffle]
    ytr = np.array(ytr)[shuffle]
    Xva = np.array(Xva).astype('f4')
    yva = np.array(yva)
    Xte = np.array(Xte).astype('f4')
    yte = np.array(yte)
    pdb.set_trace()
    with open(data_folder + data_version + '_Xtr.npy', 'wb') as f:
        np.save(f, Xtr)
    with open(data_folder + data_version + '_Xva.npy', 'wb') as f:
        np.save(f, Xva)
    with open(data_folder + data_version + '_Xte.npy', 'wb') as f:
        np.save(f, Xte)
    with open(data_folder + data_version + '_ys.pkl', 'wb') as f:
        pickle.dump((ytr, yva, yte, classes), f)
