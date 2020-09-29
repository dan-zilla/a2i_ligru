import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def metrics(hist, pte, yte, classes, epochs, str_summary):
    # https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    fig, (ax1, ax2, axc) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.plot(range(epochs), hist['TrLoss'], 'ro-', label='Training')
    ax1.plot(range(epochs), hist['VaLoss'], 'bo-', label='Validation')
    # ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 2)
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')

    ax2.plot(range(epochs), hist['TrAcc'], 'ro-', label='Training')
    ax2.plot(range(epochs), hist['VaAcc'], 'bo-', label='Validation')
    # ax1.set_xlim(0, 100)
    ax2.set_ylim(0.5, 1)
    ax2.grid(True)
    ax2.legend()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.text(epochs * 0.2, 0.5, str_summary)

    conf_te = confusion_matrix(yte, pte)
    im = axc.imshow(np.log(1 + conf_te), cmap='jet')
    axc.set_xticks(range(len(classes)))
    axc.set_yticks(range(len(classes)))
    axc.set_xticklabels(classes)
    axc.set_yticklabels(classes)
    plt.setp(axc.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = axc.text(j, i, conf_te[i, j], ha='center', va='center', color='w')
    axc.set_title('Confusion Matrix')
    axc.grid(False)

    fig.tight_layout()
    # fig.savefig()
    # plt.show()


def run_eval(model, Xev, yev, batch_size, device, obj):
    yp = []
    cur_loss = 0
    cur_acc = 0
    model.eval()
    for step in range((Xev.shape[0] + batch_size - 1) // batch_size):
        x = torch.from_numpy(Xev[step * batch_size: (step + 1) * batch_size]).float().to(device)
        y = torch.from_numpy(yev[step * batch_size: (step + 1) * batch_size]).to(device)
        with torch.no_grad():
            y_pred = model(x)
        yp.append(y_pred.cpu().numpy())
        cur_loss += obj(y_pred, y).item()
        cur_acc += np.sum((y_pred.max(dim=1)[1] == y).cpu().numpy())
    return np.argmax(np.vstack(yp), -1), cur_loss / Xev.shape[0], cur_acc / Xev.shape[0]


def lrf(epoch):
    # Cosine Annealing: https://arxiv.org/pdf/1608.03983.pdf
    jumps = [0, 20, 50, 100]
    for i in range(1, len(jumps)):
        if epoch < jumps[i]:
            return (1 + np.cos(np.pi * (epoch - jumps[i - 1]) / (jumps[i] - jumps[i - 1]))) / 2
    return 1e-6


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1) - 1, -1, -1), ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]


def quantize_with_gain(frame_data, B, gain, llim=-1, ulim=1, log_gain=False):
    bins = (((ulim-llim) / 2 ** B) * np.arange(0, 2 ** B)) + llim
    reconstr_frame = sum(frame_data)
    mean_frame = np.mean(frame_data, axis=1)
    gained_data = frame_data + 20 * np.log10(gain) if log_gain else frame_data * gain
    quant_frame_data = ((ulim-llim) / 2 ** B) * np.digitize(gained_data.T, bins) + llim
    mean_quant = np.mean(quant_frame_data, axis=0)
    quant_data = quant_frame_data - mean_quant + mean_frame  # maintain the mean (to avoid a biased estimator?)
    # ------------------------# for debug
    # x = quant_filt_frame
    # y = quant_frames[:, :, i] # for debug
    # z = filt_frame
    # x_reconstructed = sum(x.T)
    # y_reconstructed = sum(y.T)
    # z_reconstructed = sum(z)
    # plt.figure()
    # plt.plot(x_reconstructed)
    # plt.plot(y_reconstructed)
    # plt.plot(z_reconstructed)
    # plt.plot(frames[i, :])
    # -----------------------#
    return quant_data

#
# def make_gsc_dataset(classes, unknown, train_data_dir, va, te, get_train, get_silent, get_unknown):
#     Xtr = []
#     Xva = []
#     Xte = []
#     ytr = []
#     yva = []
#     yte = []
#     for i,c in enumerate(classes):
#         print('Processing "%s"'%c)
#         if c == 'unknown':
#             for cl in unknown:
#                 print('\tProcessing "%s"'%cl)
#                 all_fnames = os.listdir(train_data_dir + cl)
#                 fnames_tr = []
#                 fnames_va = []
#                 fnames_te = []
#                 for fname in all_fnames:
#                     fhash = fname.split('_')[0]
#                     if fhash in va[cl]: fnames_va.append(fname)
#                     elif fhash in te[cl]: fnames_te.append(fname)
#                     else: fnames_tr.append(fname)
#
#                 print('On Training (total %d)'%(len(fnames_tr)))
#                 sys.stdout.flush()
#                 xsa = pool.map(get_unknown, enumerate(map(lambda fname: train_data_dir + '%s/%s'%(cl, fname), fnames_tr)))
#                 # xsa = map(get_unknown, enumerate(map(lambda fname: train_data_dir + '%s/%s' % (cl, fname), fnames_tr)))
#                 xsf = [x for xs in xsa for x in xs]
#                 Xtr += xsf
#                 ytr += [i] * len(xsf)
#                 print()
#                 for ix, fname in enumerate(fnames_va):
#                     print('\rOn Validation %d/%d'%(ix+1,len(fnames_va)), end='')
#                     if np.random.uniform() > 1/20: continue
#                     Xva.append(conv_wav_to_img(get_wav(train_data_dir + '%s/%s'%(cl, fname))))
#                     yva.append(i)
#                 print()
#                 for ix, fname in enumerate(fnames_te):
#                     print('\rOn Testing %d/%d'%(ix+1,len(fnames_te)), end='')
#                     if np.random.uniform() > 1/20: continue
#                     Xte.append(conv_wav_to_img(get_wav(train_data_dir + '%s/%s'%(cl, fname))))
#                     yte.append(i)
#                 print()
#         elif c == 'silent':
#             ltr = len(Xtr) // 11
#             lva = len(Xva) // 11
#             lte = len(Xte) // 11
#             print('On Training (total %d)'%(ltr))
#             sys.stdout.flush()
#             xsa = pool.map(get_silent, range(ltr))
#             xsf = [x for xs in xsa for x in xs]
#             Xtr += xsf
#             ytr += [i] * len(xsf)
#             print()
#             for j in range(lva):
#                 print('\rOn validation %d/%d    '%(j+1,lva), end='')
#                 Xva.append(conv_wav_to_img(get_random_background()))
#                 yva.append(i)
#             print()
#             for j in range(lte):
#                 print('\rOn testing %d/%d       '%(j+1,lte), end='')
#                 Xte.append(conv_wav_to_img(get_random_background()))
#                 yte.append(i)
#             print()
#         else:
#             all_fnames = os.listdir(train_data_dir + c)
#             fnames_tr = []
#             fnames_va = []
#             fnames_te = []
#             for fname in all_fnames:
#                 fhash = fname.split('_')[0]
#                 if fhash in va[c]: fnames_va.append(fname)
#                 elif fhash in te[c]: fnames_te.append(fname)
#                 else: fnames_tr.append(fname)
#
#             print('On Training (total %d)'%(size_multiplier * len(fnames_tr)))
#             sys.stdout.flush()
#             xsa = pool.map(get_train, enumerate(map(lambda fname: train_data_dir + '%s/%s'%(c, fname), fnames_tr)))
#             # xsa = map(get_train, enumerate(map(lambda fname: train_data_dir + '%s/%s'%(c, fname), fnames_tr)))
#             xsf = [x for xs in xsa for x in xs]
#             Xtr += xsf
#             ytr += [i] * len(xsf)
#             print()
#             for ix, fname in enumerate(fnames_va):
#                 print('\rOn Validation %d/%d'%(ix+1,len(fnames_va)), end='')
#                 Xva.append(conv_wav_to_img(get_wav(train_data_dir + '%s/%s'%(c, fname))))
#                 yva.append(i)
#             print()
#             for ix, fname in enumerate(fnames_te):
#                 print('\rOn Testing %d/%d'%(ix+1,len(fnames_te)), end='')
#                 Xte.append(conv_wav_to_img(get_wav(train_data_dir + '%s/%s'%(c, fname))))
#                 yte.append(i)
#             print()
#         print()
#
#     # Final processing, shuffling, saving.
#     shuffle = np.arange(len(Xtr))
#     np.random.shuffle(shuffle)
#     Xtr = (np.array(Xtr).astype('f4'))[shuffle]
#     ytr = np.array(ytr)[shuffle]
#     Xva = np.array(Xva).astype('f4')
#     yva = np.array(yva)
#     Xte = np.array(Xte).astype('f4')
#     yte = np.array(yte)
#     with open(data_folder + data_version + '_Xtr.npy', 'wb') as f:
#         np.save(f, Xtr)
#     with open(data_folder + data_version + '_Xva.npy', 'wb') as f:
#         np.save(f, Xva)
#     with open(data_folder + data_version + '_Xte.npy', 'wb') as f:
#         np.save(f, Xte)
#     with open(data_folder + data_version + '_ys.pkl', 'wb') as f:
#         pickle.dump((ytr, yva, yte, classes), f)