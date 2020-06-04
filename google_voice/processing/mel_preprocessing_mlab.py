from __future__ import absolute_import, division, print_function
import os, sys, pdb, pickle
from multiprocessing import Pool
import numpy as np
import numpy.matlib
import scikits.samplerate as samplerate
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt
import csv
import pdb
np.seterr(all='raise')

# from sets import Set

# INPUT DIRECTORIES
# train_data_dir = '/data/workspace/danvilla/speech_commands/speech_commands_v0.01_tdmodels/'
train_data_dir = '/data/workspace/danvilla/speech_commands/chip16_spectrograms/'
test_data_dir = '../test_real/'
sub_data_dir = '../test_sub/audio/'

# OUTPUT INFORMATION
data_folder = '../outfiles_chip/'
data_version = 'processed-mlab_chip16_arduino'

#### Standard Spectrogram #####
fnames = [
    "yes/0a7c2a8d_nohash_0-0.csv",
    "no/0a9f9af7_nohash_0-0.csv",
    "yes/0c5027de_nohash_1-0.csv",
    "no/0bde966a_nohash_1-0.csv",
    "yes/0ff728b5_nohash_2-0.csv",
    "no/0ff728b5_nohash_2-0.csv",
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

########## Mel Specrogram ############
sample_rate = 16000
fs = sample_rate
frame_size = 0.025
frame_stride = 0.01
frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))

frame_duration = 1 # for now, we'll force all spectrograms to be exactly 1 sec long
num_frames =  (frame_duration / frame_stride) - 2

NFFT = 512
nfilt = 32
lf = 0
hf = (sample_rate / 2)
low_freq_mel  = (2595 * np.log10(1 + lf / 700))
high_freq_mel = (2595 * np.log10(1 + hf / 700))  # Convert Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / sample_rate)
fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1])   # left
    f_m = int(bin[m])             # center
    f_m_plus = int(bin[m + 1])    # right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

######## Background Noise for Data Augmentation #######
bg_noise = [ (train_data_dir + '_background_noise_/%s'%fname) for fname in os.listdir(train_data_dir + '_background_noise_') if fname.split('.')[-1] == 'csv' ]
def get_random_background():
    filename = bg_noise[np.random.choice(len(bg_noise))]    # len=6
    spectrogram = conv_csv_to_img(filename)
    if np.random.uniform() < 0.1:
        return np.zeros((nfilt, spectrogram.shape[1]))
    return spectrogram.astype('f4')

#for fname in fnames:
#    dat = conv_wav_to_img_noisy(wav.read(train_data_dir + fname)[1])
#    dat -= np.min(dat)
#    dat /= np.max(dat)
#    print(fname)
#    plt.imshow(dat, interpolation='nearest', origin='lower')
#    plt.show()

######### Import Data From CSV ###########
def conv_csv_to_img(filename):
    spectrogram = []
    # num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame
    if '\0' in open(filename).read():
        print("wtf " + filename)
    with open(filename, 'rt') as csv_file:
        csv_obj = csv.reader(csv_file)
        for row in csv_obj:
            spectrogram.append(row)
    # add columns to make frames all be same length
    extra_frames = int(round(len(spectrogram[0]) - num_frames))
    spectrogram = (np.array(spectrogram)).astype('f4')
    # if (extra_frames < -20):
    #     print("short clip")
    if (extra_frames < 0):
        spectrogram = np.append(spectrogram, np.matlib.repmat(spectrogram[:,-1],-extra_frames,1).T + 3*(2*np.random.random((spectrogram.shape[0],-extra_frames))-1), axis=1) # add 3dB of noise to the
    elif (extra_frames > 0):
        spectrogram = spectrogram[:,0:num_frames]

    return spectrogram.astype('f4')


def conv_csv_to_img_aug(data):
    # Get the features
    img = conv_csv_to_img(data)

    # Frequency shifting
    # orig_shape = img.shape
    fshift = np.random.randint(-2, 3)
    if fshift < 0:
        std = np.std(img[-4:], 0)
        img = np.vstack([img[-fshift:], img[-1] + std * np.random.randn(-fshift, img.shape[-1])])
    if fshift > 0:
        std = np.std(img[:4], 0)
        img = np.vstack([img[0] + std * np.random.randn(fshift, img.shape[-1]), img[:-fshift]])
    return img.astype('f4')

plt.figure(figsize=(20, 12))
for i in range(6):
    plt.subplot(3,2,i+1)
    plt.imshow(conv_csv_to_img(train_data_dir + fnames[i]), interpolation='nearest', origin='lower')
    plt.title('Mel ' + fnames[i])
plt.show()

########## Main Code for Generating Training Dataset ##########
np.random.seed(0)

vals_raw = open(train_data_dir + 'validation_list_mlab.txt', 'r').readlines()
va = {}
for fname in vals_raw:
    fsplit = fname.replace('/', '_').split('_')
    if fsplit[0] not in va: va[fsplit[0]] = set()
    va[fsplit[0]].add(fsplit[1])

tests_raw = open(train_data_dir + 'testing_list_mlab.txt', 'r').readlines()
te = {}
for fname in tests_raw:
    fsplit = fname.replace('/', '_').split('_')
    if fsplit[0] not in te: te[fsplit[0]] = set()
    te[fsplit[0]].add(fsplit[1])

size_multiplier = 5
unknown = ['off', 'on', 'right', 'stop', 'up', 'down', 'go', 'left', 'seven', 'cat', 'four', 'three', 'zero', 'tree', 'eight', 'six', 'bird', 'one', 'two', 'five', 'house', 'marvin', 'happy', 'nine', 'bed', 'sheila', 'wow', 'dog' ]
# unknown = ['unknown', 'yes', 'stop', 'no', 'go', 'seven', 'cat', 'four', 'three', 'zero', 'tree', 'eight', 'six', 'bird', 'one', 'two', 'five', 'house', 'marvin', 'happy', 'nine', 'bed', 'sheila', 'wow', 'dog' ]
# classes = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes', 'unknown', 'silent']
# classes = ['down',  'left', 'off', 'on', 'right', 'up', 'silent']
classes = ['no', 'yes', 'unknown', 'silent']

def get_train(x):
    i, fname = x
    print('\rOn %d'%(i+1), end='')
    sys.stdout.flush()
    xs = []
    xs.append(conv_csv_to_img(fname))
    for j in range(size_multiplier - 1):
        xs.append(conv_csv_to_img(fname))
    return xs

def get_unknown(x):
    i, fname = x
    print('\rOn %d'%(i+1), end='')
    sys.stdout.flush()
    pcur = size_multiplier / 20
    xs = []
    if np.random.uniform() > pcur: return xs
    xs.append(conv_csv_to_img(fname))
    pcur -= 1
    while np.random.uniform() <= pcur:
        xs.append(conv_csv_to_img(fname))
        pcur -= 1
    return xs

def get_silent(i):
    print('\rOn %d'%(i+1), end='')
    sys.stdout.flush()
    xs = []
    xs.append(get_random_background())
    #if np.random.uniform() < 0.1: xs.append(conv_wav_to_img(get_random_background()))
    #else: xs.append(conv_wav_to_img_noisy(get_random_background()))
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
        if c == 'unknown':
            for cl in unknown:
                print('\tProcessing "%s"'%cl)
                all_fnames = [('%s'%fname) for fname in os.listdir(train_data_dir + cl) if fname.split('.')[-1] == 'csv']
                fnames_tr = []
                fnames_va = []
                fnames_te = []
                for fname in all_fnames:
                    fhash = fname.split('_')[0]
                    if fhash in va[cl]: fnames_va.append(fname)
                    elif fhash in te[cl]: fnames_te.append(fname)
                    else: fnames_tr.append(fname)
                
                print('On Training (total %d)'%(len(fnames_tr)))
                sys.stdout.flush()
                xsa = pool.map(get_unknown, enumerate(map(lambda fname: train_data_dir + '%s/%s'%(cl, fname), fnames_tr)))
                xsf = [x for xs in xsa for x in xs]
                Xtr += xsf
                ytr += [i] * len(xsf)
                print()
                for ix, fname in enumerate(fnames_va):
                    print('\rOn Validation %d/%d'%(ix+1,len(fnames_va)), end='')
                    if np.random.uniform() > 1/20: continue
                    Xva.append(conv_csv_to_img(train_data_dir + '%s/%s'%(cl, fname)))
                    yva.append(i)
                print()
                for ix, fname in enumerate(fnames_te):
                    print('\rOn Testing %d/%d'%(ix+1,len(fnames_te)), end='')
                    if np.random.uniform() > 1/20: continue
                    Xte.append(conv_csv_to_img(train_data_dir + '%s/%s'%(cl, fname)))
                    yte.append(i)
                print()
        elif c == 'silent':
            ltr = len(Xtr) // 11
            lva = len(Xva) // 11
            lte = len(Xte) // 11
            print('On Training (total %d)'%(ltr))
            sys.stdout.flush()
            xsa = pool.map(get_silent, range(ltr))
            xsf = [x for xs in xsa for x in xs]
            Xtr += xsf
            ytr += [i] * len(xsf)
            print()
            for j in range(lva):
                print('\rOn validation %d/%d    '%(j+1,lva), end='')
                Xva.append(get_random_background())
                yva.append(i)
            print()
            for j in range(lte):
                print('\rOn testing %d/%d       '%(j+1,lte), end='')
                Xte.append(get_random_background())
                yte.append(i)
            print()
        else:
            all_fnames = [('%s'%fname) for fname in os.listdir(train_data_dir + c) if fname.split('.')[-1] == 'csv']
            fnames_tr = []
            fnames_va = []
            fnames_te = []
            for fname in all_fnames:
                fhash = fname.split('_')[0]
                if fhash in va[c]: fnames_va.append(fname)
                elif fhash in te[c]: fnames_te.append(fname)
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
                print('\rOn Testing %d/%d'%(ix+1,len(fnames_te)), end='')#[i]*len(xsf)
                Xte.append(conv_csv_to_img(train_data_dir + '%s/%s'%(c, fname)))
                yte.append(i)
            print()
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
    # pdb.set_trace()
    with open(data_folder + data_version + '_Xtr.npy', 'wb') as f:
        np.save(f, Xtr)
    with open(data_folder + data_version + '_Xva.npy', 'wb') as f:
        np.save(f, Xva)
    with open(data_folder + data_version + '_Xte.npy', 'wb') as f:
        np.save(f, Xte)
    with open(data_folder + data_version + '_ys.pkl', 'wb') as f:
        pickle.dump((ytr, yva, yte, classes), f)
