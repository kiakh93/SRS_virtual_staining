# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:41:28 2020

@author: kf4
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May 14 19:33:21 2020

@author: kf4
"""

import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.networks.base_network import BaseNetwork
from IPython.core.debugger import set_trace


class MultiscaleDiscriminator(BaseNetwork):


    def __init__(self):
        super().__init__()

        for i in range(2):
            subnetD = self.create_single_discriminator()
            self.add_module('discriminator_%d' % i, subnetD)

    def create_single_discriminator(self):
        subarch = 'n_layer'
        if subarch == 'n_layer':
            netD = NLayerDiscriminator()
        else:
            raise ValueError('unrecognized discriminator subarchitecture %s' % subarch)
        return netD

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3,
                            stride=2, padding=[1, 1],
                            count_include_pad=False)

    # Returns list of lists of discriminator outputs.
    # The final result is of size opt.num_D x opt.n_layers_D
    def forward(self, input):
        result = []
        get_intermediate_features = True
        for name, D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)

        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(BaseNetwork):

    def __init__(self):
        super().__init__()

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = 64
#        input_nc = self.compute_D_input_nc()

#        norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        sequence = [[nn.Conv2d(3, nf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, False)]]

        for n in range(1, 4):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == 4 - 1 else 2
            sequence += [[nn.InstanceNorm2d(nf, affine=False),(nn.Conv2d(nf_prev, nf, kernel_size=kw,
                                               stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)
                          ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))


    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        get_intermediate_features = True
        if get_intermediate_features:
            return results[1:]
        else:
            return results[-1]
