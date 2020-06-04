import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def metrics(hist, pte, yte, classes, epochs, str_summary):
    # https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html
    fig, (ax1, ax2, axc) = plt.subplots(1,3,figsize=(18, 6))

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
    ax2.text(epochs*0.2, 0.5, str_summary)

    conf_te = confusion_matrix(yte, pte)
    im = axc.imshow(np.log(1 + conf_te), cmap='jet')
    axc.set_xticks(range(len(classes)))
    axc.set_yticks(range(len(classes)))
    axc.set_xticklabels(classes)
    axc.set_yticklabels(classes)
    plt.setp(axc.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = axc.text(j,i,conf_te[i,j], ha='center', va='center', color='w')
    axc.set_title('Confusion Matrix')
    axc.grid(False)

    fig.tight_layout()
    # fig.savefig()
    # plt.show()


def run_eval(model, Xev, yev, batch_size, device, obj):
    yp = []
    cur_loss = 0
    cur_acc  = 0
    model.eval()
    for step in range((Xev.shape[0] + batch_size - 1) // batch_size):
        x = torch.from_numpy(Xev[step * batch_size : (step + 1) * batch_size]).float().to(device)
        y = torch.from_numpy(yev[step * batch_size : (step + 1) * batch_size]).to(device)
        with torch.no_grad():
            y_pred = model(x)
        yp.append(y_pred.cpu().numpy())
        cur_loss += obj(y_pred, y).item()
        cur_acc  += np.sum((y_pred.max(dim=1)[1] == y).cpu().numpy())
    return np.argmax(np.vstack(yp), -1), cur_loss / Xev.shape[0], cur_acc / Xev.shape[0]


def lrf(epoch):
    ''' Cosine Annealing: https://arxiv.org/pdf/1608.03983.pdf '''
    jumps = [0, 20, 50, 100]
    for i in range(1, len(jumps)):
        if epoch < jumps[i]: return (1 + np.cos(np.pi * (epoch - jumps[i-1]) / (jumps[i] - jumps[i-1]))) / 2
    return 1e-6


def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]