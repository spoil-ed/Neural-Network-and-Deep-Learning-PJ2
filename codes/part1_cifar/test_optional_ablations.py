import torch

from models import ConvNet, count_parameters
from run_ablations import configs


def test_convnet_optional_knobs_preserve_output_shape():
    x = torch.randn(2, 3, 32, 32)
    variants = [
        ConvNet(pooling='avg'),
        ConvNet(depth_variant='deep'),
        ConvNet(classifier_width=256),
        ConvNet(classifier_width=1024),
        ConvNet(norm='group'),
        ConvNet(conv_dropout=0.1),
    ]
    for model in variants:
        model.eval()
        with torch.no_grad():
            y = model(x)
        assert y.shape == (2, 10)


def test_convnet_optional_knobs_change_parameterization():
    base = ConvNet()
    deep = ConvNet(depth_variant='deep')
    narrow_head = ConvNet(classifier_width=256)
    wide_head = ConvNet(classifier_width=1024)
    assert count_parameters(deep) > count_parameters(base)
    assert count_parameters(narrow_head) < count_parameters(base)
    assert count_parameters(wide_head) > count_parameters(base)


def test_new_optional_ablation_groups_are_defined():
    arch = configs('optional_arch')
    training = configs('optional_training')
    assert len(arch) >= 8
    assert len(training) >= 7
    assert all('model_kw' in cfg for _, cfg in arch)
    assert all('loader_kw' in cfg for _, cfg in training)
