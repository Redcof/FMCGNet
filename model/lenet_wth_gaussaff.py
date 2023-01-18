import torch
import torch.nn as nn


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
        self.num_classes = num_classes
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
        self.fc3 = nn.Linear(fc2o, num_classes)

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
        self.softmax = nn.Softmax(dim=num_classes)

    def forward(self, x):
        # Perform the forward pass of the LeNet architecture
        x = self.pool(self.activation(self.conv1(x)))
        x = self.pool(self.activation(self.conv2(x)))
        x = x.view(-1, self.fx * self.fx * self.f2)  # flatten
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        # x = self.softmax(x)

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
    def __init__(self, sigma=1.0):
        super(GaussianAffinityLoss, self).__init__()
        self.sigma = sigma

    def forward(self, embeddings, labels):
        n = embeddings.size(0)
        dist = torch.pow(embeddings, 2).sum(1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, embeddings, embeddings.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)
        y = dist_an.new().resize_as_(dist_an).fill_(1)
        loss = torch.log(1 + torch.exp(self.sigma * (dist_ap - dist_an)))
        return loss.mean()
