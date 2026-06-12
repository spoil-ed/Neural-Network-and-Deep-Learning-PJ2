"""Models for the CIFAR-10 experiments.

Two families:
  * ConvNet  -- a compact, fully configurable VGG-style network used for the
                controlled ablations (width / activation / batch-norm / dropout).
                Contains the four *required* components: 2D conv, 2D pooling,
                fully-connected layers and activations.
  * ResNet18 -- a CIFAR-adapted residual network used as the headline model.
                Contains the optional *residual connection* and *batch-norm*
                components and reaches the best test accuracy.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model):
    return int(sum(np.prod(p.shape) for p in model.parameters()))


def make_activation(name):
    name = name.lower()
    return {
        'relu': nn.ReLU(inplace=True),
        'leaky_relu': nn.LeakyReLU(0.1, inplace=True),
        'gelu': nn.GELU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
    }[name]


def make_norm(name, channels):
    if name == 'none':
        return None
    if name == 'batch':
        return nn.BatchNorm2d(channels)
    if name == 'group':
        groups = min(32, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError(f'unknown norm {name}')


def make_pool(name):
    if name == 'max':
        return nn.MaxPool2d(2, 2)
    if name == 'avg':
        return nn.AvgPool2d(2, 2)
    raise ValueError(f'unknown pooling {name}')


# --------------------------------------------------------------------------- #
#  Configurable VGG-style ConvNet (used for ablations)                        #
# --------------------------------------------------------------------------- #
class ResStage(nn.Module):
    """Conv stage wrapped with a 1x1 projection shortcut (optional-components
    ablation): out = act(main(x) + shortcut(x)) followed by max-pooling."""

    def __init__(self, cin, cout, n_convs, activation, norm, pooling,
                 conv_dropout):
        super().__init__()
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1))
            norm_layer = make_norm(norm, cout)
            if norm_layer is not None:
                layers.append(norm_layer)
            if i < n_convs - 1:
                layers.append(make_activation(activation))
                if conv_dropout > 0:
                    layers.append(nn.Dropout2d(conv_dropout))
        self.main = nn.Sequential(*layers)
        sc = [nn.Conv2d(cin, cout, 1)]
        norm_layer = make_norm(norm, cout)
        if norm_layer is not None:
            sc.append(norm_layer)
        self.shortcut = nn.Sequential(*sc)
        self.act = make_activation(activation)
        self.drop = nn.Dropout2d(conv_dropout) if conv_dropout > 0 else nn.Identity()
        self.pool = make_pool(pooling)

    def forward(self, x):
        return self.pool(self.drop(self.act(self.main(x) + self.shortcut(x))))


class ConvNet(nn.Module):
    """A small VGG-style network with knobs for the ablation study.

    Args:
        width_mult: channel multiplier (controls #filters / #neurons).
        activation: one of relu / leaky_relu / gelu / tanh / sigmoid.
        use_bn:     insert BatchNorm after every conv.
        dropout:    dropout probability in the classifier (0 disables it).
        residual:   wrap every conv stage with a projection shortcut.
        pooling:    max or average pooling after each stage.
        depth_variant: default or deep (one extra conv per stage).
        classifier_width: hidden units in the classifier.
        norm:       none / batch / group; defaults from use_bn for backwards
                    compatibility.
        conv_dropout: Dropout2d probability inside convolutional stages.
    """

    # base channel counts of the five conv stages
    base = [64, 128, 256, 512]

    def __init__(self, num_classes=10, width_mult=1.0, activation='relu',
                 use_bn=True, dropout=0.0, residual=False, pooling='max',
                 depth_variant='default', classifier_width=512, norm=None,
                 conv_dropout=0.0):
        super().__init__()
        c = [max(8, int(round(b * width_mult))) for b in self.base]
        if norm is None:
            norm = 'batch' if use_bn else 'none'
        if depth_variant not in ('default', 'deep'):
            raise ValueError(f'unknown depth_variant {depth_variant}')
        self.act_name = activation
        self.use_bn = (norm == 'batch')
        self.norm = norm

        def block(cin, cout, n_convs):
            layers = []
            for i in range(n_convs):
                layers.append(nn.Conv2d(cin if i == 0 else cout, cout, 3, padding=1))
                norm_layer = make_norm(norm, cout)
                if norm_layer is not None:
                    layers.append(norm_layer)
                layers.append(make_activation(activation))
                if conv_dropout > 0:
                    layers.append(nn.Dropout2d(conv_dropout))
            layers.append(make_pool(pooling))
            return layers

        stage_spec = [(3, c[0], 1), (c[0], c[1], 1), (c[1], c[2], 2), (c[2], c[3], 2)]
        if depth_variant == 'deep':
            stage_spec = [(cin, cout, n + 1) for cin, cout, n in stage_spec]
        if residual:
            self.features = nn.Sequential(
                *[ResStage(cin, cout, n, activation, norm, pooling,
                           conv_dropout)
                  for cin, cout, n in stage_spec])
        else:
            self.features = nn.Sequential(
                *[layer for cin, cout, n in stage_spec
                  for layer in block(cin, cout, n)])
        feat_dim = c[3] * 2 * 2
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, classifier_width),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(classifier_width, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def features_forward(self, x):
        """Return flattened penultimate features (for t-SNE / interpretation)."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        for layer in self.classifier[:-1]:
            x = layer(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# --------------------------------------------------------------------------- #
#  CIFAR ResNet-18 (headline model)                                           #
# --------------------------------------------------------------------------- #
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut = nn.Sequential()
        if stride != 1 or cin != cout * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(cout * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)   # residual connection
        return F.relu(out)


class ResNet18(nn.Module):
    """ResNet-18 adapted for 32x32 inputs (3x3 stem, no initial max-pool)."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _make_layer(self, planes, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def _embed(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, 1)
        return torch.flatten(out, 1)

    def features_forward(self, x):
        return self._embed(x)

    def forward(self, x):
        return self.fc(self._embed(x))


# --------------------------------------------------------------------------- #
#  WideResNet (Zagoruyko & Komodakis, 2016) -- the accuracy-push model         #
# --------------------------------------------------------------------------- #
class WRNBlock(nn.Module):
    def __init__(self, cin, cout, stride, drop=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.drop = drop
        self.equal = (cin == cout and stride == 1)
        self.shortcut = None if self.equal else nn.Conv2d(cin, cout, 1, stride, 0, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        s = x if self.equal else self.shortcut(o)
        o = self.conv1(o)
        o = F.relu(self.bn2(o))
        if self.drop > 0:
            o = F.dropout(o, self.drop, self.training)
        return self.conv2(o) + s


class WideResNet(nn.Module):
    """WRN-(depth)-(widen). Pre-activation residual blocks, no bottleneck."""

    def __init__(self, depth=28, widen=10, num_classes=10, drop=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        w = [16, 16 * widen, 32 * widen, 64 * widen]
        self.conv1 = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        self.group1 = self._group(w[0], w[1], n, 1, drop)
        self.group2 = self._group(w[1], w[2], n, 2, drop)
        self.group3 = self._group(w[2], w[3], n, 2, drop)
        self.bn = nn.BatchNorm2d(w[3])
        self.fc = nn.Linear(w[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    @staticmethod
    def _group(cin, cout, n, stride, drop):
        layers = [WRNBlock(cin, cout, stride, drop)]
        layers += [WRNBlock(cout, cout, 1, drop) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def _embed(self, x):
        o = self.conv1(x)
        o = self.group3(self.group2(self.group1(o)))
        o = F.relu(self.bn(o))
        return torch.flatten(F.adaptive_avg_pool2d(o, 1), 1)

    def features_forward(self, x):
        return self._embed(x)

    def forward(self, x):
        return self.fc(self._embed(x))


def build_model(name, **kwargs):
    name = name.lower()
    if name == 'resnet18':
        return ResNet18(**kwargs)
    if name == 'convnet':
        return ConvNet(**kwargs)
    if name == 'wrn28_10':
        return WideResNet(depth=28, widen=10, **kwargs)
    if name == 'wrn28_10_drop':
        return WideResNet(depth=28, widen=10, drop=0.3, **kwargs)
    raise ValueError(f'unknown model {name}')


if __name__ == '__main__':
    for n, m in [('ConvNet', ConvNet()), ('ResNet18', ResNet18())]:
        print(f'{n:10s} params = {count_parameters(m):,}')
