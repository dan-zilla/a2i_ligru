from __future__ import absolute_import, division, print_function
import os, sys, pdb, pickle
from multiprocessing import Pool
import numpy as np
import scikits.samplerate as samplerate
import scipy.io.wavfile as wav
import scipy.signal as sig
import matplotlib.pyplot as plt
import a2i_helper as a2i
np.seterr(all='raise')

# from sets import Set

# INPUT DIRECTORIES
train_data_dir = '/data/speech_commands_v0.01/'
test_data_dir = '../test/'
sub_data_dir = '../test_sub/audio/'

# OUTPUT INFORMATION
data_folder = '../outfiles/'
rms = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
rms = [1.6]
bits = [2, 3, 10]
bits = [10]
data_version = 'processed-12_mel_noiseless_larger'

#### Standard Spectrogram #####
fnames = [
    "yes/0a7c2a8d_nohash_0.wav",
    "no/0a9f9af7_nohash_0.wav",
    "yes/0c5027de_nohash_1.wav",
    "no/0bde966a_nohash_1.wav",
    "yes/0ff728b5_nohash_2.wav",
    "no/0ff728b5_nohash_2.wav",
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

# Automatic Gain Control
TARGET_RMS = 3.2 # of full-scale = 1
MAX_GAIN = 1000
NORM_RMS = True
B = 6 # quantization bits
QUANT_FILT = True
QUANT_SIG = False
def conv_wav_to_img(data):
    '''
    Assumes data is in the [-1,1] range (from wav read)
    '''
    data = data[:sample_rate].astype('f4')
    data -= np.mean(data)
    data_rms = np.sqrt(np.sum(data ** 2 / len(data))) + np.finfo(float).eps
    gain = min((TARGET_RMS / data_rms), MAX_GAIN) if NORM_RMS else 1

    if len(data) < fs: data = np.pad(data, (0, fs - len(data)), 'edge')
    
    signal_length = len(data)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.finfo(float).eps * np.ones((pad_signal_length - signal_length))
    pad_signal = np.append(data, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    frames = (frames.T - np.mean(frames, axis=1)).T * np.hanning(frame_length)

    fft_frames = (1.0 / NFFT) * np.fft.rfft(frames, NFFT)

    if QUANT_FILT:
        quant_frames = np.finfo(float).eps * np.ones(((fft_frames.shape[1] - 1) * 2, nfilt, fft_frames.shape[0]))
        quant_pow_frames = np.zeros((fft_frames.shape[0], nfilt))

        for i in range(fft_frames.shape[0]):
            filt_frame_fft = NFFT * fft_frames[i,:] * fbank
            filt_frame = np.fft.irfft(filt_frame_fft)
            quant_frames[:, :, i] = a2i.quantize_with_gain(filt_frame, B, gain)

            quant_frame_fft = np.fft.rfft(quant_frames[:, :, i], axis=0) / NFFT
            quant_pow_frames[i,:] = np.sum((np.absolute(quant_frame_fft))**2, 0)
        filter_banks = quant_pow_frames
    else:
        mag_frames = np.absolute(fft_frames)  # Magnitude of the FFT
        pow_frames = mag_frames ** 2  # Power Spectrum
        filter_banks = np.dot(pow_frames, fbank.T)
        # bins = ((1 / 2 ** B) * np.arange(0, 2 ** B)) - 1
        # filter_banks = np.digitize(filter_banks, bins)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 10 * np.log10(filter_banks)  # dB
    #filter_banks = np.sqrt(filter_banks)
    filter_banks = filter_banks[2:]
    return filter_banks.astype('f4').T


def get_wav(file):
    waveform_int = wav.read(file)[1] # / 2**15
    # clip_rms = np.sqrt(np.sum(waveform**2) / len(waveform))
    # waveform = target_rms / clip_rms * waveform
    # https://stackoverflow.com/questions/47762432/fast-way-to-quantize-numpy-vectors
    waveform = (waveform_int & -0b100000) / 2**15 if QUANT_SIG else waveform_int / 2 ** 15 # TODO: make this programmatic with B
    return waveform


plt.figure(figsize=(20, 12))
for i in range(6):
    # i=1 # debug
    plt.subplot(3,2,i+1)
    plt.imshow(conv_wav_to_img(get_wav(train_data_dir + fnames[i])), interpolation='nearest', origin='lower')
    plt.title('Mel ' + fnames[i])
plt.savefig('spectrograms.pdf')
plt.show()

######## Background Noise for Data Augmentation #######
bg_noise = [ get_wav(train_data_dir + '_background_noise_/%s'%fname) for fname in os.listdir(train_data_dir + '_background_noise_') if fname.split('.')[-1] == 'wav' ]
def get_random_background():
    if np.random.uniform() < 0.1: return np.zeros(fs)
    data = bg_noise[np.random.choice(len(bg_noise))]
    start = np.random.randint(0, len(data) - fs)
    return data[start:start+fs]

def conv_wav_to_img_noisy(data):
    # Time clipping and dilation
    clip_begin = np.random.randint(3000)
    clip_end   = np.random.randint(1, 3000)
    data_clip = data[clip_begin:-clip_end]
    data_resamp = samplerate.resample(data_clip, np.random.uniform(0.5,1.5), 'sinc_best') # https://stackoverflow.com/questions/29085268/resample-a-numpy-array/29085482
    if len(data_resamp) > fs:
        start = (len(data_resamp) - fs) // 2
        data_resamp = data_resamp[start:start+fs]
    
    # Add background noise
    data_full = get_random_background()
    data_add = data_resamp
    snr = np.var(data_add) / (1 + np.var(data_full))
    data_full = np.sqrt(0.1 * snr) * np.random.uniform() * data_full
    start = 0 if fs == len(data_add) else np.random.randint(fs - len(data_add))
    data_full[start:start+len(data_add)] += data_add * sig.tukey(len(data_add), 0.25)
    
    # Amplitude scaling
    data_full *= np.exp(np.random.randn())
    
    # Get the features
    img = conv_wav_to_img(data_full)
    
    # Frequency shifting
    orig_shape = img.shape
    fshift = np.random.randint(-2,3)
    if fshift < 0:
        std = np.std(img[-4:], 0)
        img = np.vstack([img[-fshift:], img[-1] + std * np.random.randn(-fshift, img.shape[-1])])
    if fshift > 0:
        std = np.std(img[:4], 0)
        img = np.vstack([img[0] + std * np.random.randn(fshift, img.shape[-1]), img[:-fshift]])
    return img.astype('f4')

#for fname in fnames:
#    dat = conv_wav_to_img_noisy(wav.read(train_data_dir + fname)[1])
#    dat -= np.min(dat)
#    dat /= np.max(dat)
#    print(fname)
#    plt.imshow(dat, interpolation='nearest', origin='lower')
#    plt.show()

########## Main Code for Generating Training Dataset ##########
np.random.seed(0)

vals_raw = open(train_data_dir + 'validation_list.txt', 'r').readlines()
va = {}
for fname in vals_raw:
    fsplit = fname.replace('/', '_').split('_')
    if fsplit[0] not in va: va[fsplit[0]] = set()
    va[fsplit[0]].add(fsplit[1])

tests_raw = open(train_data_dir + 'testing_list.txt', 'r').readlines()
te = {}
for fname in tests_raw:
    fsplit = fname.replace('/', '_').split('_')
    if fsplit[0] not in te: te[fsplit[0]] = set()
    te[fsplit[0]].add(fsplit[1])

size_multiplier = 1
unknown = ['seven', 'cat', 'four', 'three', 'zero', 'tree', 'eight', 'six', 'bird', 'one', 'two', 'five', 'house', 'marvin', 'happy', 'nine', 'bed', 'sheila', 'wow', 'dog' ]
classes = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes', 'unknown', 'silent']

def get_train(x):
    i, fname = x
    print('\rOn %d'%(i+1), end='')
    sys.stdout.flush()
    xs = []
    waveform = get_wav(fname)
    xs.append(conv_wav_to_img(waveform))
    for j in range(size_multiplier - 1):
        xs.append(conv_wav_to_img_noisy(waveform))
    return xs

def get_unknown(x):
    i, fname = x
    print('\rOn %d'%(i+1), end='')
    sys.stdout.flush()
    # pcur = size_multiplier / 20
    pcur = 1
    xs = []
    if np.random.uniform() > pcur: return xs
    waveform = get_wav(fname)
    xs.append(conv_wav_to_img(waveform))
    pcur -= 1
    while np.random.uniform() <= pcur:
        xs.append(conv_wav_to_img_noisy(waveform))
        pcur -= 1
    return xs

def get_silent(i):
    print('\rOn %d'%(i+1), end='')
    sys.stdout.flush()
    xs = []
    xs.append(conv_wav_to_img(get_random_background()))
    #if np.random.uniform() < 0.1: xs.append(conv_wav_to_img(get_random_background()))
    #else: xs.append(conv_wav_to_img_noisy(get_random_background()))
    return xs

if __name__ == "__main__":
    pool = Pool(12)
    for k in rms:
        TARGET_RMS = k
        for j in bits:
            B = j
            # data_version = 'processed-12_mel_noiseless_'+str(B)+'bit_quant_env_'+str(TARGET_RMS).replace('.','p')+'rms'
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
                        all_fnames = os.listdir(train_data_dir + cl)
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
                        # xsa = pool.map(get_unknown, enumerate(map(lambda fname: train_data_dir + '%s/%s'%(cl, fname), fnames_tr)))
                        xsa = map(get_unknown, enumerate(map(lambda fname: train_data_dir + '%s/%s' % (cl, fname), fnames_tr)))
                        xsf = [x for xs in xsa for x in xs]
                        Xtr += xsf
                        ytr += [i] * len(xsf)
                        print()
                        for ix, fname in enumerate(fnames_va):
                            print('\rOn Validation %d/%d'%(ix+1,len(fnames_va)), end='')
                            if np.random.uniform() > 1/20: continue
                            Xva.append(conv_wav_to_img(get_wav(train_data_dir + '%s/%s'%(cl, fname))))
                            yva.append(i)
                        print()
                        for ix, fname in enumerate(fnames_te):
                            print('\rOn Testing %d/%d'%(ix+1,len(fnames_te)), end='')
                            if np.random.uniform() > 1/20: continue
                            Xte.append(conv_wav_to_img(get_wav(train_data_dir + '%s/%s'%(cl, fname))))
                            yte.append(i)
                        print()
                elif c == 'silent':
                    ltr = len(Xtr) // 11
                    lva = len(Xva) // 11
                    lte = len(Xte) // 11
                    print('On Training (total %d)'%(ltr))
                    sys.stdout.flush()
                    # xsa = pool.map(get_silent, range(ltr))
                    xsa = map(get_silent, range(ltr))
                    xsf = [x for xs in xsa for x in xs]
                    Xtr += xsf
                    ytr += [i] * len(xsf)
                    print()
                    for j in range(lva):
                        print('\rOn validation %d/%d    '%(j+1,lva), end='')
                        Xva.append(conv_wav_to_img(get_random_background()))
                        yva.append(i)
                    print()
                    for j in range(lte):
                        print('\rOn testing %d/%d       '%(j+1,lte), end='')
                        Xte.append(conv_wav_to_img(get_random_background()))
                        yte.append(i)
                    print()
                else:
                    all_fnames = os.listdir(train_data_dir + c)
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
                    # xsa = pool.map(get_train, enumerate(map(lambda fname: train_data_dir + '%s/%s'%(c, fname), fnames_tr)))
                    xsa = map(get_train, enumerate(map(lambda fname: train_data_dir + '%s/%s'%(c, fname), fnames_tr)))
                    xsf = [x for xs in xsa for x in xs]
                    Xtr += xsf
                    ytr += [i] * len(xsf)
                    print()
                    for ix, fname in enumerate(fnames_va):
                        print('\rOn Validation %d/%d'%(ix+1,len(fnames_va)), end='')
                        Xva.append(conv_wav_to_img(get_wav(train_data_dir + '%s/%s'%(c, fname))))
                        yva.append(i)
                    print()
                    for ix, fname in enumerate(fnames_te):
                        print('\rOn Testing %d/%d'%(ix+1,len(fnames_te)), end='')
                        Xte.append(conv_wav_to_img(get_wav(train_data_dir + '%s/%s'%(c, fname))))
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
            with open(data_folder + data_version + '_Xtr.npy', 'wb') as f:
                np.save(f, Xtr)
            with open(data_folder + data_version + '_Xva.npy', 'wb') as f:
                np.save(f, Xva)
            with open(data_folder + data_version + '_Xte.npy', 'wb') as f:
                np.save(f, Xte)
            with open(data_folder + data_version + '_ys.pkl', 'wb') as f:
                pickle.dump((ytr, yva, yte, classes), f)
