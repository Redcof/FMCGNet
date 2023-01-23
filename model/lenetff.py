import os
import pathlib
from abc import abstractmethod

import numpy as np
import sklearn
import torch
from torch import nn, optim
from tqdm import tqdm

from model.dcn import DeformableConv2d


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


class LeNetFF(nn.Module):
    LAYER_NO = 0

    def __init__(self, sample_batch, if0, epoch, num_classes, channels=1, deformable=False, lr=0.001):
        super().__init__()
        self.num_class = num_classes
        self.layers = []
        k1, f1, s1, p1, d1 = 5, 6, 1, 0, 2
        k2, f2, s2, p2, d2 = 5, 16, 1, 0, 2
        ps = 2
        fc1o = 120
        fc2o = 84

        self.conv1 = FFConvLayer(in_channels=channels, out_channels=f1, kernel_size=k1, stride=s1, padding=p1,
                                 dilation=d1, deformable=deformable, pool_kernel=2, pool_stride=ps, lr=lr)
        self.conv2 = FFConvLayer(in_channels=f1, out_channels=f2, kernel_size=k2, stride=s2, padding=p2,
                                 dilation=d2, deformable=deformable, pool_kernel=2, pool_stride=ps, lr=lr)

        x = sample_batch
        x = self.conv1(x)
        x = self.conv2(x)
        fc_fx_size = x.shape[1] * x.shape[2] * x.shape[3]

        self.flatten = FFFlatten()
        self.fc1 = FFFCLayer(fc_fx_size, fc1o, lr=lr)
        laynorm = FFLayerNorm(1, eps=1e-5, elementwise_affine=False)
        self.fc2 = FFFCLayer(fc1o, fc2o, lr=lr)
        self.fc3 = FFFCLayer(fc2o, num_classes, lr=lr)
        # self.softmax = nn.Softmax(dim=num_classes)
        self.layers = [
            self.conv1, self.conv2, self.flatten, self.fc1, self.fc2, self.fc3
        ]

    def predict(self, x, num_classes):
        goodness_per_label = []
        for label in range(num_classes):
            h = overlay_y_on_x(x, label, num_classes)
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

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for idx, layer in enumerate(self.layers):
            # print('Layer:[%02d/%02d]...' % (idx, LeNetFF.LAYER_NO), end="\b\r")
            h_pos, h_neg = layer.train(h_pos, h_neg)


class FFLayer(nn.Module):
    def __init__(self, goodness_threshold=2.0, layer_norm=True, epoch=10, activation=None, criterion=None):
        super().__init__()
        LeNetFF.LAYER_NO += 1
        self.layer_no = LeNetFF.LAYER_NO
        self.threshold = goodness_threshold
        self.num_epochs = epoch
        self.layer_norm = layer_norm
        self.activation = torch.nn.ReLU() if activation is None else activation
        self.criterion = self.calculate_loss if criterion is None else criterion

    @abstractmethod
    def goodness(self, x):
        return None

    def calculate_loss(self, f_pos, f_neg):
        goodness_pos = self.goodness(f_pos)
        goodness_neg = self.goodness(f_neg)
        # The following loss pushes pos (neg) samples to
        # values larger (smaller) than the self.threshold.
        loss = torch.log(1 + torch.exp(torch.cat([
            -goodness_pos + self.threshold,
            goodness_neg - self.threshold]))).mean()
        return loss

    def train(self, x_pos, x_neg):
        for epoch_idx in range(self.num_epochs):
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


class FFFlatten(nn.Flatten):
    def __int__(self, *args, **kwargs):
        super(FFFlatten, self).__int__(*args, **kwargs)

    def goodness(self, x):
        return None

    def train(self, x_pos, x_neg):
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFLayerNorm(nn.LayerNorm):
    def __int__(self, *args, **kwargs):
        super(FFLayerNorm, self).__int__(*args, **kwargs)

    def goodness(self, x):
        return x.pow(2).mean(dim=(1,))  # positive mean square as goodness

    def train(self, x_pos, x_neg):
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()


class FFFCLayer(FFLayer):
    def __init__(self, in_features, out_features, dtype=None,
                 layer_norm=True, activation=None, goodness_threshold=2.0, epoch=10, optimizer=None, lr=0.0001,
                 criterion=None):
        super().__init__(goodness_threshold=goodness_threshold, layer_norm=layer_norm, epoch=epoch,
                         activation=activation, criterion=criterion)
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
                 criterion=None):
        super().__init__(goodness_threshold=goodness_threshold, layer_norm=layer_norm, epoch=epoch,
                         activation=activation, criterion=criterion)
        self.fc = nn.Linear(in_features, out_features, dtype=dtype)
        self.optimizer = optim.Adam(self.fc.parameters(), lr=lr) if optimizer is None else optimizer

    def forward(self, x):
        x = self.activation(self.fc(x))
        return x

    def goodness(self, x):
        return x.pow(2).mean(dim=(1,))  # positive mean square as goodness


class FFFCOrigLayer(FFLayer):
    def __init__(self, in_features, out_features, dtype=None,
                 layer_norm=True, activation=None, goodness_threshold=2.0, epoch=10, optimizer=None, lr=0.0001,
                 criterion=None, classification_criterion=None):
        super().__init__(goodness_threshold=goodness_threshold, layer_norm=layer_norm, epoch=epoch,
                         activation=activation, criterion=criterion)
        self.fc = nn.Linear(in_features, out_features, dtype=dtype)
        self.optimizer = optim.Adam(self.fc.parameters(), lr=lr) if optimizer is None else optimizer
        self.criterion = nn.CrossEntropyLoss() if classification_criterion is None else classification_criterion

    def calculate_loss(self, f_pos, f_neg, y_label):
        goodness_pos = self.goodness(f_pos)
        goodness_neg = self.goodness(f_neg)
        # The following loss pushes pos (neg) samples to
        # values larger (smaller) than the self.threshold.
        loss = torch.log(1 + torch.exp(torch.cat([
            -goodness_pos + self.threshold,
            goodness_neg - self.threshold]))).mean()
        # additional classification loss
        class_loss_value = self.criterion(f_pos, y_label)
        return loss + class_loss_value

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
                 criterion=None):
        super().__init__(goodness_threshold=goodness_threshold, layer_norm=layer_norm, epoch=epoch,
                         activation=activation, criterion=criterion)
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


def evaluate(writer, net, opt, test_loader, classes, op_dir, batch_idx):
    with torch.no_grad():
        print("\n\tTesting...")
        acc_list = []
        fpr_list = []
        f1_list = []
        pred_y = []
        actual_y = []
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

            acc_list.append(acc)
            fpr_list.append(fpr)
            f1_list.append(f1)

            writer.add_scalar('Testing Accuracy[%s]' % classes[class_idx], acc, batch_idx)
            writer.add_scalar('Testing F1-Score[%s]' % classes[class_idx], f1, batch_idx)
            writer.add_scalar('Testing FPR[%s]' % classes[class_idx], fpr, batch_idx)

        writer.add_scalar('Testing mean Accuracy', np.mean(acc_list), batch_idx)
        writer.add_scalar('Testing mean F1-Score', np.mean(f1_list), batch_idx)
        writer.add_scalar('Testing mean FPR', np.mean(fpr_list), batch_idx)

        with open(str(op_dir / "model_summary.txt"), "a+") as fp:
            acc_list.append(np.mean(acc_list))
            f1_list.append(np.mean(f1_list))
            fpr_list.append(np.mean(fpr_list))
            s = 'Testing Accuracy: %s\n' % acc_list
            fp.write(s)
            print(s.strip())
            s = 'Testing F1-Score: %s\n' % f1_list
            fp.write(s)
            print(s.strip())
            s = 'Testing FPR: %s\n' % fpr_list
            fp.write(s)
            print(s.strip())


def train_loop(opt, classes, writer, train_loader, test_loader, val_loader):
    num_batches = len(train_loader)
    sample_batch = torch.rand((opt.batchsize, opt.nc, opt.isize, opt.isize))
    net = LeNetFF(sample_batch=sample_batch, if0=opt.niter, num_classes=len(classes), epoch=opt.niter, channels=3,
                  deformable=opt.deformable)
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
    for batch_idx, ((batch_x, batch_y), meta) in enumerate(tqdm(train_loader)):
        # print("Batch: [%d/%d]" % (idx, num_batches))
        x_pos = overlay_y_on_x_batch(batch_x, batch_y, len(classes))
        rnd = torch.randperm(batch_x.size(0))
        x_neg = overlay_y_on_x_batch(batch_x, batch_y[rnd], len(classes))
        # x_pos = x_pos.view(-1, 3 * 128 * 128)  # flatten(but batch wise)
        # x_neg = x_neg.view(-1, 3 * 128 * 128)  # flatten(but batch wise)
        net.train(x_pos, x_neg)

        # unlike backprop, in FF we evaluate per-batch (not per-epoch)
        evaluate(writer, net, opt, test_loader, classes, op_dir, batch_idx)


def train_loop2(opt, classes, writer, train_loader, test_loader, val_loader):
    from forward_forward import train_with_forward_forward_algorithm
    import os
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trained_model = train_with_forward_forward_algorithm(
        model_type="progressive",
        n_layers=3,
        hidden_size=2000,
        lr=0.0001,
        device=device,
        epochs=100,
        batch_size=opt.batchsize,
        theta=2.,
    )
