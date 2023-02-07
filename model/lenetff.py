import os
import pathlib
from abc import abstractmethod
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from PIL import Image
from empatches import EMPatches
from matplotlib import pyplot as plt
from torch import nn, optim
from torchvision.transforms import transforms
from tqdm import tqdm

from model.dcn import DeformableConv2d
from options import Options

"""
How to create a FF layer?

Any FF layer class must contain the following:

Properties:
    batch_idx
    loss
    goodness_val

Methods:
        def goodness(self, x):...
        def train(x_pos, x_neg):...
        def calculate_loss(self, f_pos, f_neg,):...
Sample:
class FFCustomLayer:
    def __int__(self, *args, **kwargs):
        self.loss = None
        self.goodness_val = None
        LeNetFF.LAYER_NO += 1
        self.layer_no = LeNetFF.LAYER_NO
        self.batch_idx = 0

        def goodness(self, x):
            return None

        def train(self, x_pos, x_neg):
            return self.forward(x_pos).detach(), self.forward(x_neg).detach()
        
        def calculate_loss(self, f_pos, f_neg):
            goodness_pos = self.goodness(f_pos)
            goodness_neg = self.goodness(f_neg)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -goodness_pos + self.threshold,
                goodness_neg - self.threshold]))).mean()
            self.loss = loss.items()
            self.goodness_val = goodness_pos.mean().item()
            return loss
"""


def get_batch_of_hybrid_image(x_batch, y_batch, opt, classes):
    hybrid_b = torch.rand_like(x_batch).cpu()
    for batch_idx, cls_idx in enumerate(y_batch):
        if batch_idx > 4:
            # assuming every last to 3rd item form now is from different class
            ineg = x_batch[batch_idx - 3]
        else:
            # assuming every next to 3rd item form now is from different class
            ineg = x_batch[batch_idx + 3]
        ineg = ineg.clone().detach().cpu().numpy().reshape((opt.isize, opt.isize, opt.nc))
        ipos = x_batch[batch_idx].clone().detach().cpu().numpy().reshape((opt.isize, opt.isize, opt.nc))
        npimage = hybrid_negative_image(opt, classes[cls_idx], ipos=ipos, ineg=ineg)
        pil = Image.fromarray(npimage)
        npimage = np.array(opt.transform(pil))
        hybrid_b[batch_idx, :, :, :] = torch.from_numpy(npimage.reshape((opt.nc, opt.isize, opt.isize)))
    return hybrid_b.to(opt.device)


def hybrid_negative_image(opt, class_lbl, ipos=None, ineg=None):
    """
    From original paper: "The Forward-Forward Algorithm: Some Preliminary Investigations"
    To force FF to focus on the longer range correlations in images that characterize shapes, we need
to create negative data that has very different long range correlations but very similar short range
correlations. This can be done by creating a mask containing fairly large regions of ones and zeros.
We then create hybrid images for the negative data by adding together one digit image times the mask
and a different digit image times the reverse of the mask as shown in figure 1. Masks like this can be
created by starting with a random bit image and then repeatedly blurring the image with a filter of the
form [1/4, 1/2, 1/4] in both the horizontal and vertical directions. After repeated blurring, the image
is then threshold at 0.5.

    1. get two images(ipos, ineg) one for given class and another form any ransom class
    2. create random bit image (ir)
    3. apply motion blur(vertical and horizontal) 3 times (ibl)
    4. apply threshold(0.5) (imsk, irmsk)
    4. hybrid = ipos*imsk + ineg*irmsk


    Returns: negative hybrid image
    """

    def bluring(img, kernel_size):
        # Specify the kernel size.
        # The greater the size, the more the motion.

        # Create the vertical kernel.
        kernel_v = np.zeros((kernel_size, kernel_size))

        # Create a copy of the same for creating the horizontal kernel.
        kernel_h = np.copy(kernel_v)

        # Fill the middle row with ones.
        kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)

        # Normalize.
        kernel_v /= kernel_size
        kernel_h /= kernel_size

        # Apply the vertical kernel.
        img = cv2.filter2D(img, -1, kernel_v)

        # Apply the horizontal kernel.
        img = cv2.filter2D(img, -1, kernel_h)
        return img

    def get_patch(sample, opt):
        img_path = sample['image'].values[0]
        patch_id = sample['patch_id'].values[0]
        img_path = os.path.join(opt.dataroot, img_path)
        patch_size = opt.isize
        patch_overlap = opt.atz_patch_overlap
        image = cv2.imread(img_path)
        # create patches
        emp = EMPatches()
        img_patches, indices = emp.extract_patches(image, patchsize=patch_size, overlap=patch_overlap)
        img = img_patches[patch_id]
        return img

    if ipos is None or ineg is None:
        patch_dataset_csv = opt.atz_patch_db
        df = pd.read_csv(patch_dataset_csv)
        if ipos is None:
            sample1 = df[(df['label_txt'] == class_lbl) & (df['anomaly_size'] >= opt.isize ** 2 / 2)].sample(1)
            ipos = get_patch(sample1, opt)

        if ineg is None:
            sample2 = df[(df['label_txt'] != class_lbl) & (df['anomaly_size'] >= 0.0)].sample(1)
            ineg = get_patch(sample2, opt)

    ipos = cv2.cvtColor(ipos, cv2.COLOR_RGB2GRAY)
    ineg = cv2.cvtColor(ineg, cv2.COLOR_RGB2GRAY)

    ipos = cv2.normalize(ipos, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ineg = cv2.normalize(ineg, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    np.random.seed(opt.manualseed)
    ir = np.random.rand(*ipos.shape)
    ir = cv2.normalize(ir, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ibl = bluring(bluring(bluring(ir, 40), 40), 40)

    _, imsk = cv2.threshold(ibl, 255 // 2, 255, cv2.THRESH_BINARY)
    _, irmsk = cv2.threshold(ibl, 255 // 2, 255, cv2.THRESH_BINARY_INV)

    hybrid = (ipos * imsk) + (ineg * irmsk)

    # images = [ir, ibl, ipos, imsk, hybrid, irmsk, ineg]
    # title = ["random", "blured", "pos", "* mask", "+", "(1-mask)*", "neg"]
    # cols = len(images)
    # for idx, img in enumerate(images):
    #     ax = plt.subplot(1, cols, idx + 1)
    #     ax.set_title(title[idx])
    #     plt.imshow(img, cmap="Greys_r")
    #     # cv2.imshow(title[idx], img)
    # plt.show()
    hybrid = cv2.cvtColor(hybrid, cv2.COLOR_GRAY2RGB)
    return hybrid


def overlay_y_on_x_batch(x_batch, y_batch, n_classes):
    x_batch = x_batch.clone()
    for idx, (x, y) in enumerate(zip(x_batch, y_batch)):
        x = overlay_y_on_x(x, y, n_classes, inplace=True)
        x_batch[idx, :, :, :] = x
    return x_batch


def overlay_y_on_x(x, y, n_classes, inplace=False):
    if inplace is False:
        x = x.clone()
    x[:, :n_classes] *= 0.0
    try:
        x[range(x.shape[0]), y] = x.max()
    except Exception as e:
        print("=>>", x.shape, range(x.shape[0]), y, torch.tensor(0, dtype=torch.uint8), x.max())
        raise e
    return x


# ################################################################################
# ################################################################################
# ################################################################################
# ################################################################################
# ################################################################################
class LeNetFF(nn.Module):
    LAYER_NO = 0

    def __init__(self, sample_batch, if0, epoch, num_classes, channels=1, deformable=False, lr=0.001, writer=None,
                 goodness=2.0, negative_image_algo="overlay", batchnorm=False, layernorm=False, dropout=False):
        super().__init__()
        self.num_class = num_classes
        self.layers = []
        self.negative_image_algo = negative_image_algo
        k1, f1, s1, p1, d1 = 5, 6, 1, 0, 2
        k2, f2, s2, p2, d2 = 5, 16, 1, 0, 2
        ps = 2
        fc1o = 1024
        fc2o = 512
        fc3o = 256

        self.conv1 = FFConvLayer(in_channels=channels, out_channels=f1, kernel_size=k1, stride=s1, padding=p1,
                                 epoch=epoch,
                                 goodness_threshold=goodness,
                                 dilation=d1, deformable=deformable, pool_kernel=2, pool_stride=ps, lr=lr,
                                 writer=writer)
        self.batch_norm1 = FFBatchNorm2d(num_features=f1, affine=False) if batchnorm else FFForwardLayer()
        self.conv2 = FFConvLayer(in_channels=f1, out_channels=f2, kernel_size=k2, stride=s2, padding=p2, epoch=epoch,
                                 goodness_threshold=goodness,
                                 dilation=d2, deformable=deformable, pool_kernel=2, pool_stride=ps, lr=lr,
                                 writer=writer)
        self.batch_norm2 = FFBatchNorm2d(num_features=f2, affine=False) if batchnorm else FFForwardLayer()

        x = sample_batch
        x = self.conv1(x)
        x = self.conv2(x)
        conv_op_size = x.shape[1] * x.shape[2] * x.shape[3]

        self.flatten = FFFlatten()

        self.fc1 = FFFCLayer(conv_op_size, fc1o, lr=lr, goodness_threshold=goodness, writer=writer, epoch=epoch)
        self.drop1 = FFDropout(p=0.2) if dropout else FFForwardLayer()
        self.laynorm1 = FFLayerNorm(fc1o, eps=1e-5, elementwise_affine=False) if layernorm else FFForwardLayer()

        self.fc2 = FFFCLayer(fc1o, fc2o, lr=lr, goodness_threshold=goodness, writer=writer, epoch=epoch)
        self.drop2 = FFDropout(p=0.2) if dropout else FFForwardLayer()
        self.laynorm2 = FFLayerNorm(fc2o, eps=1e-5, elementwise_affine=False) if layernorm else FFForwardLayer()

        self.fc3 = FFFCLayer(fc2o, fc3o, lr=lr, goodness_threshold=goodness, writer=writer, epoch=epoch)
        self.drop3 = FFDropout(p=0.2) if dropout else FFForwardLayer()
        self.laynorm3 = FFLayerNorm(fc3o, eps=1e-5, elementwise_affine=False) if layernorm else FFForwardLayer()

        self.fc4 = FFFCLayer(fc3o, num_classes, lr=lr, goodness_threshold=goodness, writer=writer, epoch=epoch)

        # compose layers
        self.layers = [self.conv1, self.batch_norm1, self.conv2, self.batch_norm2,
                       self.flatten,
                       self.fc1, self.laynorm1, self.drop1,
                       self.fc2, self.laynorm2, self.drop2,
                       self.fc3, self.laynorm3, self.drop3,
                       self.fc4
                       ]

    def predict(self, x, num_classes):
        goodness_per_label = []
        for label in range(num_classes):
            if self.negative_image_algo == "overlay":
                h = overlay_y_on_x(x, label, num_classes)
            elif self.negative_image_algo == "hybrid":
                h = x  # do nothing
            h = h.view(1, *h.shape)  # reshape to batch of 1 image
            goodness = []
            for layer in self.layers:
                h = layer(h)
                gdns = layer.goodness(h)  # as 'h' is a batch of 1 image, take first item
                if gdns is not None:
                    smile = gdns[0]
                    goodness += [smile]
            goodness_per_label += [sum(goodness).unsqueeze(0)]
        goodness_per_label = torch.cat(goodness_per_label, 0)
        return goodness_per_label.argmax(0), h

    def train(self, x_pos, x_neg, batch_idx):
        h_pos, h_neg = x_pos, x_neg
        for idx, layer in enumerate(self.layers):
            print('\tTraining Layer:[%d/%d]...' % (idx, LeNetFF.LAYER_NO))
            layer.batch_idx = batch_idx
            h_pos, h_neg = layer.train(h_pos, h_neg)
            try:
                print("\tLayer-", layer.layer_no, "Loss:", layer.loss, "Goodness:", layer.goodness_val)
            except:
                ...


class FFLayer(nn.Module):
    def __init__(self, goodness_threshold=2.0, layer_norm=True, epoch=10, activation=None, criterion=None, writer=None):
        super().__init__()
        LeNetFF.LAYER_NO += 1
        self.layer_no = LeNetFF.LAYER_NO
        self.threshold = goodness_threshold
        self.num_epochs = epoch
        self.layer_norm = layer_norm
        self.writer = writer
        self.batch_idx = 0
        self.activation = torch.nn.ReLU() if activation is None else activation
        self.criterion = self.calculate_loss if criterion is None else criterion
        self.loss = float("-inf")
        self.goodness_val = 0.0

    @abstractmethod
    def goodness(self, x):
        return None

    def calculate_loss(self, f_pos, f_neg):
        goodness_pos = self.goodness(f_pos)
        goodness_neg = self.goodness(f_neg)
        # The following loss pushes pos (neg) samples to
        # values larger (smaller) than the self.threshold.
        val = torch.cat([-goodness_pos + self.threshold, goodness_neg - self.threshold])
        loss = torch.log(1 + torch.exp(val)).mean()
        if self.writer:
            self.writer.add_scalars('Goodness', {"Layer-%d" % self.layer_no: goodness_pos.mean()},
                                    self.batch_idx + 1)
            self.writer.add_scalars('Loss', {"Layer-%d" % self.layer_no: loss}, self.batch_idx + 1)
        self.loss = loss.item()
        self.goodness_val = goodness_pos.mean().item()
        return loss

    def train(self, x_pos, x_neg):
        for epoch_idx in tqdm(range(self.num_epochs)):
            f_pos = self.forward(x_pos)
            f_neg = self.forward(x_neg)
            # calculate loss
            loss = self.criterion(f_pos, f_neg)
            self.optimizer.zero_grad()  # set gradients to 0
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()  # backprop loss in layer
            self.optimizer.step()  # update weights
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFFCLayer(FFLayer):
    def __init__(self, in_features, out_features, dtype=None,
                 layer_norm=True, activation=None, goodness_threshold=2.0, epoch=2, optimizer=None, lr=0.0001,
                 criterion=None, writer=None):
        super().__init__(goodness_threshold=goodness_threshold, layer_norm=layer_norm, epoch=epoch,
                         activation=activation, criterion=criterion, writer=writer)
        self.fc = nn.Linear(in_features, out_features, dtype=dtype)
        self.optimizer = optim.Adam(self.fc.parameters(), lr=lr) if optimizer is None else optimizer

    def forward(self, x):
        x = self.activation(self.fc(x))
        return x

    def goodness(self, x):
        return x.pow(2).mean(dim=(1,))  # positive mean square as goodness


class FFClassificationLayer(FFLayer):
    def __init__(self, in_features, out_features, dtype=None,
                 layer_norm=True, activation=None, goodness_threshold=2.0, epoch=10, optimizer=None, lr=0.0001,
                 criterion=None, writer=None, classification_criterion=None):
        super().__init__(goodness_threshold=goodness_threshold, layer_norm=layer_norm, epoch=epoch,
                         activation=activation, criterion=criterion, writer=writer)
        self.fc = nn.Linear(in_features, out_features, dtype=dtype)
        self.optimizer = optim.Adam(self.fc.parameters(), lr=lr) if optimizer is None else optimizer
        self.criterion = nn.CrossEntropyLoss() if classification_criterion is None else classification_criterion

    def forward(self, x):
        x = self.activation(self.fc(x))
        return x

    def goodness(self, x):
        return x.pow(2).mean(dim=(1,))  # positive "mean square" as goodness

    def calculate_loss(self, f_pos, f_neg, y_label=None):
        goodness_pos = self.goodness(f_pos)
        goodness_neg = self.goodness(f_neg)
        # The following loss pushes pos (neg) samples to
        # values larger (smaller) than the self.threshold.
        val = torch.cat([-goodness_pos + self.threshold, goodness_neg - self.threshold])
        loss = torch.log(1 + torch.exp(val)).mean()
        # additional classification loss
        class_loss_value = self.criterion(f_pos, y_label)
        return loss + class_loss_value


class FFFCOrigLayer(FFLayer):
    def __init__(self, in_features, out_features, dtype=None,
                 layer_norm=True, activation=None, goodness_threshold=2.0, epoch=10, optimizer=None, lr=0.0001,
                 criterion=None, writer=None):
        super().__init__(goodness_threshold=goodness_threshold, layer_norm=layer_norm, epoch=epoch,
                         activation=activation, criterion=criterion, writer=writer)
        self.fc = nn.Linear(in_features, out_features, dtype=dtype)
        self.optimizer = optim.Adam(self.fc.parameters(), lr=lr) if optimizer is None else optimizer

    def forward(self, x):
        x_direction = x / x.norm(2, 1, keepdim=True) + 1e-4
        x = self.activation(self.fc(x_direction))
        return x

    def goodness(self, x):
        return x.pow(2).mean(dim=(1,))  # positive mean square as goodness


class FFConvLayer(FFLayer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, deformable=False,
                 pool_kernel=None, pool_stride=None,
                 activation=None, goodness_threshold=2.0, layer_norm=False, epoch=10, optimizer=None, lr=0.0001,
                 criterion=None, writer=None):
        super().__init__(goodness_threshold=goodness_threshold, layer_norm=layer_norm, epoch=epoch,
                         activation=activation, criterion=criterion, writer=writer)
        # selection of convolution
        conv = DeformableConv2d if deformable else nn.Conv2d
        self.conv = conv(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                         stride=stride, padding=padding, dilation=dilation)
        self.pool = None
        if pool_kernel is not None:
            self.pool = nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_stride)  # non-learning layer, we can reuse
        self.optimizer = optim.Adam(self.conv.parameters(), lr=lr) if optimizer is None else optimizer

    def forward(self, x):
        x = self.activation(self.conv(x))
        if self.pool is not None:
            x = self.pool(x)
        if self.layer_norm:
            x = x.norm(2, 1, keepdim=True) + 1e-4
        return x

    def goodness(self, x):
        return x.pow(2).mean(dim=(-1, -2, -3))  # positive mean square as goodness


class FFFlatten(nn.Flatten):
    def __int__(self, *args, **kwargs):
        super(FFFlatten, self).__int__(*args, **kwargs)
        LeNetFF.LAYER_NO += 1
        self.layer_no = LeNetFF.LAYER_NO
        self.batch_idx = 0
        self.loss = None
        self.goodness = None

    def goodness(self, x):
        return None

    def train(self, x_pos, x_neg):
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFLayerNorm(nn.LayerNorm):
    def __int__(self, *args, **kwargs):
        super(FFLayerNorm, self).__int__(*args, **kwargs)
        LeNetFF.LAYER_NO += 1
        self.layer_no = LeNetFF.LAYER_NO
        self.batch_idx = 0
        self.batch_idx = 0
        self.loss = None
        self.goodness = None
        self.layer_no = None

    def goodness(self, x):
        return None

    def train(self, x_pos, x_neg):
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFForwardLayer(nn.Module):
    """
    This layer does nothing to the input
    """

    def __int__(self, *args, **kwargs):
        super(FFForwardLayer, self).__int__(*args, **kwargs)
        LeNetFF.LAYER_NO += 1
        self.layer_no = LeNetFF.LAYER_NO
        self.batch_idx = 0
        self.batch_idx = 0
        self.loss = None
        self.goodness = None
        self.layer_no = None

    def goodness(self, x):
        return None

    def train(self, x_pos, x_neg):
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFDropout(nn.Dropout):
    def __int__(self, *args, **kwargs):
        super(FFDropout, self).__int__(*args, **kwargs)
        LeNetFF.LAYER_NO += 1
        self.layer_no = LeNetFF.LAYER_NO
        self.batch_idx = 0
        self.batch_idx = 0
        self.loss = None
        self.goodness = None
        self.layer_no = None

    def goodness(self, x):
        return None

    def train(self, x_pos, x_neg):
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFBatchNorm1d(nn.BatchNorm1d):
    def __int__(self, *args, **kwargs):
        super(FFBatchNorm1d, self).__int__(*args, **kwargs)
        LeNetFF.LAYER_NO += 1
        self.layer_no = LeNetFF.LAYER_NO
        self.batch_idx = 0
        self.batch_idx = 0
        self.loss = None
        self.goodness = None
        self.layer_no = None

    def goodness(self, x):
        return None

    def train(self, x_pos, x_neg):
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFBatchNorm2d(nn.BatchNorm2d):
    def __int__(self, *args, **kwargs):
        super(FFBatchNorm2d, self).__int__(*args, **kwargs)
        LeNetFF.LAYER_NO += 1
        self.layer_no = LeNetFF.LAYER_NO
        self.batch_idx = 0
        self.batch_idx = 0
        self.loss = None
        self.goodness = None
        self.layer_no = None

    def goodness(self, x):
        return None

    def train(self, x_pos, x_neg):
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFBatchNorm3d(nn.BatchNorm3d):
    def __int__(self, *args, **kwargs):
        super(FFBatchNorm3d, self).__int__(*args, **kwargs)
        LeNetFF.LAYER_NO += 1
        self.layer_no = LeNetFF.LAYER_NO
        self.batch_idx = 0
        self.batch_idx = 0
        self.loss = None
        self.goodness = None
        self.layer_no = None

    def goodness(self, x):
        return None

    def train(self, x_pos, x_neg):
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


def evaluate(writer, net, opt, test_loader, classes, op_dir, batch_idx):
    with torch.no_grad():
        print("\n\tTesting...")
        acc_list = []
        fpr_list = []
        tpr_list = []
        f1_list = []
        pred_y = []
        actual_y = []
        tps = []
        tns = []
        fps = []
        fns = []
        for idx, ((batch_x, batch_y), meta) in enumerate(tqdm(test_loader)):
            for x, y in zip(batch_x, batch_y):
                one_goodness, y_pred = net.predict(x, len(classes))
                pred_y.append(one_goodness.item())
                actual_y.append(y.item())
        actual_y, pred_y = np.array(actual_y), np.array(pred_y)
        for class_idx in range(len(classes)):
            one_vs_rest_actual_y = actual_y == class_idx
            one_vs_rest_pred_y = pred_y == class_idx
            # evaluate
            confusion_mat = sklearn.metrics.confusion_matrix(one_vs_rest_actual_y, one_vs_rest_pred_y)
            tp, fn, fp, tn = confusion_mat.flatten()
            acc = np.sum((tp, tn)) / np.sum((tn, fp, fn, tp))
            f1 = (2 * tp) / np.sum((2 * tp, fp, fn))
            fpr = fp / np.sum((tn, fp))
            tpr = tp / np.sum((tp, fn))

            acc_list.append(acc)
            fpr_list.append(fpr)
            f1_list.append(f1)
            tpr_list.append(tpr)

            tps.append(tp)
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)

            writer.add_scalars('Accuracy', {classes[class_idx]: acc}, batch_idx + 1)
            writer.add_scalars('F1-Score', {classes[class_idx]: f1}, batch_idx + 1)
            writer.add_scalars('FPR', {classes[class_idx]: fpr}, batch_idx + 1)
            writer.add_scalars('TPR', {classes[class_idx]: tpr}, batch_idx + 1)

        writer.add_scalars('Evaluation', {
            "µAccuracy": np.mean(acc_list),
            'µF1-Score': np.mean(f1_list),
            'µFPR': np.mean(fpr_list),
            'µTPR': np.mean(tpr_list)
        }, batch_idx + 1)
        print({
            "µAccuracy": np.mean(acc_list),
            'µF1-Score': np.mean(f1_list),
            'µFPR': np.mean(fpr_list),
            'µTPR': np.mean(tpr_list)
        })

        acc_list.append(np.mean(acc_list))
        f1_list.append(np.mean(f1_list))
        fpr_list.append(np.mean(fpr_list))
        tps.append(0)
        tns.append(0)
        fps.append(0)
        fns.append(0)
        try:
            pd.DataFrame(dict(
                Accuracy=acc_list,
                F1=f1_list,
                FPR=fpr_list,
                TPR=tpr_list,
                TP=tps,
                FP=fps,
                TN=tns,
                FN=fns,
            )).to_csv(str(op_dir / "eval.csv"))
        except:
            ...
        with open(str(op_dir / "model_summary.txt"), "a+") as fp:
            s = 'Accuracy: %s\n' % acc_list
            fp.write(s)
            s = 'F1-Score: %s\n' % f1_list
            fp.write(s)
            s = 'FPR: %s\n' % fpr_list
            fp.write(s)
            s = 'TPR: %s\n' % tpr_list
            fp.write(s)
            s = 'TP: %s\n' % tps
            fp.write(s)
            s = 'TN: %s\n' % tns
            fp.write(s)
            s = 'FP: %s\n' % fps
            fp.write(s)
            s = 'FN: %s\n' % fns
            fp.write(s)


def train_loop(opt, classes, writer, train_loader, test_loader, val_loader):
    now = datetime.now()
    dt_string_start = now.strftime("%d/%m/%Y %H:%M:%S\n")
    num_batches = len(train_loader)
    sample_batch = torch.rand((opt.batchsize, opt.nc, opt.isize, opt.isize))
    net = LeNetFF(sample_batch=sample_batch, if0=opt.niter, num_classes=len(classes), epoch=opt.niter,
                  channels=opt.nc,
                  goodness=opt.ffgoodness, negative_image_algo=opt.ffnegalg,
                  batchnorm=opt.batchnorm,
                  layernorm=opt.layernorm,
                  dropout=opt.dropout,
                  lr=opt.lr, deformable=opt.deformable, writer=writer)
    print("================Model Summary===============")
    op_dir = pathlib.Path(opt.outf) / opt.name
    os.makedirs(str(op_dir), exist_ok=True)
    with open(str(op_dir / "model_summary.txt"), "w") as fp:
        s = "\n%s\n" % net
        fp.write(opt.name)
        fp.write(s)
        print(s)
    print("============================================")
    net.to(opt.device)
    print("\nTraining...")
    for batch_idx, ((batch_x, batch_y), meta) in enumerate(train_loader):
        print("Training Batch: [%d/%d]" % (batch_idx, num_batches))
        if opt.ffnegalg == "overlay":
            x_pos = overlay_y_on_x_batch(batch_x, batch_y, len(classes))
            rnd = torch.randperm(batch_x.size(0))
            x_neg = overlay_y_on_x_batch(batch_x, batch_y[rnd], len(classes))
        elif opt.ffnegalg == "hybrid":
            x_pos = batch_x
            x_neg = get_batch_of_hybrid_image(batch_x, batch_y, opt, classes)

        # write to tensorboard
        writer.add_image('Positive Images', torchvision.utils.make_grid(x_pos))
        writer.add_image('Negative Images', torchvision.utils.make_grid(x_neg))

        # x_pos = x_pos.view(-1, 3 * 128 * 128)  # flatten(but batch wise)
        # x_neg = x_neg.view(-1, 3 * 128 * 128)  # flatten(but batch wise)
        net.train(x_pos, x_neg, batch_idx)

        # unlike backprop, in FF we evaluate per-batch (not per-epoch)
        evaluate(writer, net, opt, test_loader, classes, op_dir, batch_idx)
    with open(str(op_dir / "model_summary.txt"), "a+") as fp:
        now = datetime.now()
        dt_string_end = now.strftime("%d/%m/%Y %H:%M:%S\n")
        fp.write(dt_string_start)
        fp.write(dt_string_end)
        print(dt_string_start.strip())
        print(dt_string_end.strip())
    # # save the model
    # torch.save({'epoch': epoch_idx, 'state_dict': net.state_dict()},
    #            f'{str(weight_dir)}/net_last.pth' % (epoch_idx, auc))


if __name__ == '__main__':
    opt = Options().parse()
    ipos = None
    ineg = None
    ipos = cv2.imread("sample7.png")
    ineg = cv2.imread("sample6.png")
    ipos = cv2.resize(ipos, (190, 170))
    ineg = cv2.resize(ineg, (190, 170))
    hybrid_negative_image(opt, "GA", ipos=ipos, ineg=ineg)
