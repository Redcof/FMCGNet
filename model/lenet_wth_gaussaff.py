import torch
import torch.nn as nn
from tqdm import tqdm

from customdataset.atz.dataloader import load_atz_data, CLASSE_IDX
from options import Options


class LeNet(nn.Module):
    def __init__(self, if0=32, channels=1, num_classes=2, num_anchors=0):
        """
        LeNet input image size in 32x32  1ch
        if - input feature
        f - feature, k-kernel, s-stride, p-padding
        ps-pooling stride, fci - fully connected in
        fco - fully connected out
        """
        super(LeNet, self).__init__()
        k1, f1, s1, p1, d1 = 5, 6, 1, 0, 2
        k2, f2, s2, p2, d2 = 5, 16, 1, 0, 2
        ps = 2
        fc1o = 120
        fc2o = 84
        self.f2 = f2
        # As original LeNet designed for 32x32
        # images patches, we need to calculate
        # feature map size for other image dimensions.
        # Calculating the feature map size after
        # convolution and pooling before moving
        # into fc layers
        fx = 1 + (if0 - (k1 * d1 - 1) + (2 * p1)) // s1
        fx = fx // ps
        fx = 1 + (fx - (k1 * d2 - 1) + (2 * p2)) // s2
        self.fx = fx = fx // ps
        # calculation ends.

        self.conv1 = nn.Conv2d(channels, f1, k1, s1, p1, dilation=d1)
        self.pool = nn.MaxPool2d(2, ps)  # non-learning layer, we can reuse

        self.conv2 = nn.Conv2d(f1, f2, k2, s2, p2, dilation=d2)

        self.fc1 = nn.Linear(fx * fx * f2, fc1o)
        self.fc2 = nn.Linear(fc1o, fc2o)
        self.classification_head = nn.Linear(fc2o, num_classes)

        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.conv6 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        # self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        # self.fc4 = nn.Linear(in_features=512 * 3 * 3, out_features=1024)
        # self.fc5 = nn.Linear(in_features=1024, out_features=num_anchors * 4)  # 4 for bounding box regression
        # self.fc6 = nn.Linear(in_features=1024, out_features=num_anchors * num_classes)  # for classification
        # self.num_anchors = num_anchors
        # self.num_classes = num_classes
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # Perform the forward pass of the LeNet architecture
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, self.fx * self.fx * self.f2)  # flatten
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.classification_head(x)

        # # Additional layers for object detection
        # x = self.activation(self.conv3(x))
        # x = self.activation(self.conv4(x))
        # x = self.activation(self.conv5(x))
        # x = self.activation(self.conv6(x))
        # x = self.activation(self.conv7(x))
        # x = x.view(x.size(0), -1)
        # x = self.activation(self.fc4(x))
        #
        # # Bounding box regression
        # bboxes = self.fc5(x)
        # bboxes = bboxes.view(bboxes.size(0), self.num_anchors, 4)

        # Classification
        # logits = self.fc6(x)
        # logits = logits.view(logits.size(0), self.num_anchors, self.num_classes)
        # return bboxes, logits
        return x


class GaussianAffinityLoss(nn.Module):
    def __init__(self, lamda=0.75, sigma=1.0):
        super(GaussianAffinityLoss, self).__init__()
        self.lamda = lamda
        self.sigma = sigma

    def forward(self, logits, labels):
        # Apply the softmax function
        logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
        exp_logits = torch.exp(logits)
        probs = exp_logits / torch.sum(exp_logits, dim=1, keepdim=True)

        # Compute the cross-entropy loss
        log_probs = torch.log(probs)
        one_hot_labels = torch.eye(logits.size(1))[labels]
        loss = -torch.mean(torch.sum(one_hot_labels * log_probs, dim=1))

        return loss

        # # true
        # onehot = y_true_plusone[:, :-1]
        # # pred
        # distance = y_pred_plusone[:, :-1]
        # # Diversity Regularizer
        # rw = tf.reduce_mean(y_pred_plusone[:, -1])
        #
        # # L_mm
        # d_fi_wyi = tf.reduce_sum(onehot * distance, axis=-1, keepdims=True)
        # losses = tf.maximum(self.lambd + distance - d_fi_wyi, 0.0)
        # L_mm = tf.reduce_sum(losses * (1.0 - onehot), axis=-1)  # j!=y_i
        #
        # return L_mm + rw


import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

opt = Options().parse()
# Define the LeNet architecture
C = len(CLASSE_IDX)
net = LeNet(if0=opt.isize, channels=opt.nc, num_classes=C)

# Define the Gaussian Affinity Loss function
criterion_class_loss = nn.CrossEntropyLoss()
criterion_bbox_loss = nn.SmoothL1Loss()

# Define the optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)

# Define the number of training epochs
num_epochs = opt.niter
batch_size = opt.batchsize

# Define the data loading and preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader, test_loader, val_loader = load_atz_data(opt)

# # Train the network
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for i, data in enumerate(train_loader):
#         inputs, labels = data
#         optimizer.zero_grad()
#         bboxreg, outputs = net(inputs)
#         loss = criterion_class_loss(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print('Epoch %d loss: %.3f' % (epoch + 1, running_loss / (i + 1)))
#
# print('Finished Training')

net.to(opt.device)  # move to device
net.train()  # activate training mode
# Train the network
for epoch in range(num_epochs):
    train_running_loss = 0.0
    test_running_loss = 0.0
    train_batch = 0
    test_batch = 0
    print("Training... epoch:[%d/%d]" % (epoch + 1, num_epochs))
    # train
    for idx, data in enumerate(tqdm(train_loader, leave=False, total=len(train_loader))):
        (inputs, labels), meta = data
        optimizer.zero_grad()
        # bboxes_pred, logits_pred = net(inputs)
        logits_pred = net(inputs)  # forward
        # calculate the loss for bounding box regression
        # bbox_loss_value = criterion_bbox_loss(bboxes_pred, bboxes)
        # calculate the loss for classification
        class_loss_value = criterion_class_loss(logits_pred, labels)
        # combine the loss
        # loss = bbox_loss_value + class_loss_value
        loss = class_loss_value
        # backprop
        loss.backward()
        # update weights
        optimizer.step()
        train_running_loss += loss.item()
        train_batch += 1
        # print('Batch %d/ loss: %.3f' % (idx + 1, running_loss / (idx + 1)), end="\b\r")

    print("\nTesting... epoch:[%d/%d]" % (epoch + 1, num_epochs))
    with torch.no_grad():
        # test
        for i, data in enumerate(tqdm(test_loader, leave=False, total=len(test_loader))):
            (inputs, labels), meta = data
            # bboxes_pred, logits_pred = net(inputs)
            logits_pred = net(inputs)  # forward
            # calculate the loss for bounding box regression
            # bbox_loss_value = criterion_bbox_loss(bboxes_pred, bboxes)
            # calculate the loss for classification
            class_loss_value = criterion_class_loss(logits_pred, labels)
            # combine the loss
            # loss = bbox_loss_value + class_loss_value
            loss = class_loss_value
            test_running_loss += loss.item()
            test_batch += 1
            # print('Batch %d/ loss: %.3f' % (idx + 1, running_loss / (idx + 1)), end="\b\r")

    print('\nEpoch %d: train loss: %.3f' % (epoch + 1, train_running_loss / (train_batch + 1)))
    print('Epoch %d: test loss: %.3f' % (epoch + 1, test_running_loss / (test_batch + 1)))

print('Finished Training')
