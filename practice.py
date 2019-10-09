import re
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
import numpy as np
class ClassName(object):
	"""docstring for ClassName"""
	def __init__(self, arg):
		super(ClassName, self).__init__()
		self.arg = arg

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        if self.memory_efficient and any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features

class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)

#model=_DenseBlock(3,5,5,2,0.5)

#for name,module in model.named_children():
#	print(module)

#x=np.array([[[1,1,1],[2,2,2]]])
#print(x.shape[0])
#print(x.shape)
#print(x[:,0])
#print(x[:,1])

#a=t.Tensor([[1,2,3,4,5,6,7,8],[1,2,3,4,5,6,7,8]])
#b=a.reshape(-1)
#print(a)
#print(b)

#size=np.array([1,2,3])
#aspect_ratios=np.array([1,0.5,2])
#for s,a in zip(size,aspect_ratios):
#	print('s:{0:.3f}'.format(s))
#	print('a:{0:.3f}'.format(a))
anchors_over_all_feature_maps=np.array([[0,3,3,4,4],[1,1,1,2,2,],[2,5,5,6,6],[3,7,7,8,8]])
anchors=[]
for i in range(0,1):#imagelist
	anchor_in_image=[]
	for anchors_per_feature_map in anchors_over_all_feature_maps:
		anchor_in_image.append(anchors_per_feature_map)
	anchors.append(anchor_in_image)
	print(anchors)
anchors=[(anchors_per_feature_map) for anchors_per_image in anchors]
print(anchors)