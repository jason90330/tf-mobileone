import tensorflow as tf
import numpy as np
import os
from typing import Optional, List, Tuple
from tensorflow.keras import layers, models # type: ignore

__all__ = ['MobileOne', 'mobileone', 'reparameterize_model']

class Identity(layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, input):
        return input

class SEBlock(layers.Layer):
    def __init__(
            self,
            in_channels: int,
            rd_ratio: float = 0.0625
            ) -> None:
        
        self.reduce = layers.Conv2D(
            filters=int(in_channels*rd_ratio),
            kernel_size=1,
            strides=1,
            use_bias=True)
        
        self.expand = layers.Conv2D(
            filters=in_channels,
            kernel_size=1,
            strides=1,
            use_bias=True)
        
    
    def call(self, inputs):
        """ Apply forward pass. """
        b, c, h, w = inputs.shape.as_list()
        # x = tf.nn.avg_pool2d(inputs, ksize=[h, w], strides=[h, w], padding='VALID')
        x = layers.AveragePooling2D(pool_size=(h,w), strides=(h,w))(x)
        x = self.reduce(x)
        x = layers.ReLU(x)
        x = self.expand(x)
        x = tf.keras.activations.sigmoid(x)
        x = tf.reshape(x, (-1, c, 1, 1))
        return inputs*x

class MobileOneBlock(layers.Layer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: str = 'valid',
            dilation: int = 1,
            groups: int = 1,
            inference_mode: bool = False,
            use_se: bool = False,
            num_conv_branches: int = 1
        ):
        """ Construct a MobileOneBlock module.

        :param in_channels: Number of channels in the input.
        :param out_channels: Number of channels produced by the block.
        :param kernel_size: Size of the convolution kernel.
        :param stride: Stride size.
        :param padding: Zero-padding (same: pad with kernel_size/2, valid: not pad).
        :param dilation: Kernel dilation factor.
        :param groups: Group number.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super().__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        # Check if SE-ReLU is requested
        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = Identity()
        self.activation = layers.ReLU()

        if inference_mode:
            if self.groups==1:
                self.reparam_conv = layers.Conv2D(
                    filters = out_channels,
                    kernel_size = kernel_size,
                    strides = stride,
                    padding = padding,
                    dilation_rate = dilation,
                    use_bias = True
                )
            else:
                self.reparam_conv = layers.DepthwiseConv2D(
                    kernel_size=kernel_size, 
                    strides=stride, 
                    padding=padding, 
                    use_bias=True)

        else:
            # Re-parameterizable skip connection
            # normalize same channel between batch (n,h,w,c) -> same c over all n
            if out_channels == in_channels and stride == 1:
                self.rbr_skip = layers.BatchNormalization(axis=-1)
            else:
                self.rbr_skip = None
            
            # Re-parameterizable conv branches
            self.rbr_conv = []
            for _ in range(self.num_conv_branches):
                self.rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))

            # Re-parameterizable scale branch
            self.rbr_scale = None
            if kernel_size > 1:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding='valid')

    def call(self, x):
        """ Apply forward pass. """
        # Inference mode forward pass.
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        for ix in range(self.num_conv_branches):
            out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))

    def _conv_bn(self, 
                 kernel_size: int, 
                 padding: str) -> tf.keras.Sequential:
        mod_list = []
        if self.groups==1:
            mod_list.append(
                layers.Conv2D(
                    filters = self.out_channels,
                    kernel_size = kernel_size,
                    strides = self.stride,
                    padding = padding,
                    use_bias = False,
                    name="conv"
                )
            )
        else:
            mod_list.append(
                layers.DepthwiseConv2D(
                    kernel_size=kernel_size, 
                    strides=self.stride, 
                    padding=padding,
                    use_bias=False,
                    name="conv"
                )
            )
        mod_list.append(
            layers.BatchNormalization(axis=-1, name="bn")
        )
        return tf.keras.Sequential(mod_list)
    
    def reparameterize(self):
        if self.inference_mode:
            return
        else:
            kernel, bias = self.get_equivalent_kernel_bias()
            # self.rbr_conv[0] is first branch
            # self.rbr_conv[0].layers[0] is conv
            '''
            self.reparam_conv = layers.Conv2D(
                filters = self.rbr_conv[0].layers[0].filters,
                kernel_size = self.rbr_conv[0].layers[0].kernel_size,
                strides = self.rbr_conv[0].layers[0].strides,
                padding = self.rbr_conv[0].layers[0].padding,
                dilation_rate = self.rbr_conv[0].layers[0].dilation_rate,
                use_bias = True
            )
            '''
            # method1
            # self.reparam_conv.build((1, 32, 32, self.dw_3x3_0.conv.layers[0].groups))
            
            # method2
            # self.reparam_conv.weight.data = kernel
            # self.reparam_conv.bias.data = bias

            # method3
            self.reparam_conv.set_weights([kernel, bias])

            # Delete un-used branches
            for param in self.trainable_variables:
                param._trainable = False
            if hasattr(self, 'rbr_conv'):
                del self.rbr_conv
            if hasattr(self, 'rbr_scale'):
                del self.rbr_scale
            if hasattr(self, 'rbr_skip'):
                del self.rbr_skip
        
            self.inference_mode = True
    
    def get_equivalent_kernel_bias(self):
        """ Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        :return: Tuple of (kernel, bias) after fusing branches.
        """
        kernel3x3 = kernel1x1 = kernelid = bias3x3 = bias1x1 = biasid = 0
        
        # get weights and bias of conv branches (kernel shape = h,w,cin,cout)
        for ix in range(self.num_conv_branches):
            _k3x3, _b3x3 = self._fuse_bn_tensor(self.rbr_conv[ix])
            kernel3x3 += _k3x3
            bias3x3 += _b3x3
        
        # get weights and bias of scale branch
        if self.rbr_scale is not None:
            kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_scale)# N,224,224,3 -> N,112,112,48
            kernel1x1 = self._pad_1x1_to_nxn_tensor(kernel1x1)

        # get weights and bias of skip(identity) branch
        if self.rbr_skip is not None:
            kernelid, biasid = self._fuse_bn_tensor(self.rbr_skip)
        
        #kernelsize = (ks,ks,cin,cout), eg:3*3*48*48 or 1*1*48*48
        return (
            kernel3x3 + kernel1x1 + kernelid,
            bias3x3 + bias1x1 + biasid
        )
    '''
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            if self.stride == 2:
                return tf.pad(kernel1x1, tf.constant([[0, 2], [0, 2], [0, 0], [0, 0]]), "CONSTANT")
            else:
                # padding for first dim, second dim, third dim, forth dim (eg: 1,1,3,48 -> 3,3,3,48)
                return tf.pad(kernel1x1, tf.constant([[1, 1], [1, 1], [0, 0], [0, 0]]), "CONSTANT")
    '''

    # padding for first dim, second dim, third dim, forth dim (eg: 1,1,3,48 -> 3,3,3,48)
    def _pad_1x1_to_nxn_tensor(self, x):
        pad = self.kernel_size // 2
        # x = layers.ZeroPadding2D(padding = ((pad,pad),(pad,pad)))(x)
        # return x
        return tf.pad(x, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), "CONSTANT")
    
    def _fuse_bn_tensor(self, branch):
        if isinstance(branch, tf.keras.Sequential):
            # branch.layers[0] is conv, [1] is bias
            kernel = branch.get_layer("conv").weights[0]
            moving_mean = branch.get_layer("bn").moving_mean
            moving_variance = branch.get_layer("bn").moving_variance
            gamma = branch.get_layer("bn").gamma
            beta = branch.get_layer("bn").beta
            eps = branch.get_layer("bn").epsilon
        else:
            assert isinstance(branch, layers.BatchNormalization)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.kernel_size, self.kernel_size, input_dim, self.in_channels), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[self.kernel_size//2, self.kernel_size//2, i % input_dim, i] = 1
                self.id_tensor = tf.convert_to_tensor(kernel_value, dtype=np.float32)
                # kernel_value = np.zeros((self.in_channels, input_dim, self.kernel_size, self.kernel_size), dtype=np.float32)
                # for i in range(self.in_channels):
                #     kernel_value[i, i % input_dim, self.kernel_size//2, self.kernel_size//2] = 1
                # self.id_tensor = kernel_value
            kernel = self.id_tensor
            moving_mean = branch.moving_mean
            moving_variance = branch.moving_variance
            gamma = branch.gamma
            beta = branch.beta
            eps = branch.epsilon
            if self.groups!=1:
                w,h,cin,cout = kernel.shape
                kernel = tf.reshape(kernel, (w,h,cout,cin))
        std = tf.sqrt((moving_variance + eps))
        # t = (gamma / std).reshape(-1, 1, 1, 1)
        if self.groups==1:
            t = tf.reshape((gamma / std), (1, 1, 1, -1))
        else:
            t = tf.reshape((gamma / std), (1, 1, -1, 1))
        return kernel * t, beta - moving_mean * gamma / std
    
class MobileOne(tf.keras.Model):
    def __init__(self,
                 num_blocks_per_stage: List[int] = [2, 8, 10, 1],
                 num_classes: int = 1000,
                 width_multipliers: Optional[List[float]] = None,
                 inference_mode: bool = False,
                 use_se: bool = False,
                 num_conv_branches: int = 1
            ) -> None:
        """ Construct MobileOne model.
        :param num_blocks_per_stage: List of number of blocks per stage.
        :param num_classes: Number of classes in the dataset.
        :param width_multipliers: List of width multiplier for blocks in a stage.
        :param inference_mode: If True, instantiates model in inference mode.
        :param use_se: Whether to use SE-ReLU activations.
        :param num_conv_branches: Number of linear conv branches.
        """
        super().__init__()

        assert len(width_multipliers) == 4
        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64*width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        # Build stage
        self.stage0 = MobileOneBlock(in_channels=3, out_channels=self.in_planes,
                                     kernel_size=3, stride=2, padding='same',
                                     inference_mode=self.inference_mode)
        self.cur_layer_idx = 1
        self.stage1 = self._make_stage(int(64 * width_multipliers[0]), num_blocks_per_stage[0],
                                       num_se_blocks=0)
        self.stage2 = self._make_stage(int(128 * width_multipliers[1]), num_blocks_per_stage[1],
                                       num_se_blocks=0)
        self.stage3 = self._make_stage(int(256 * width_multipliers[2]), num_blocks_per_stage[2],
                                       num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0)
        self.stage4 = self._make_stage(int(512 * width_multipliers[3]), num_blocks_per_stage[3],
                                       num_se_blocks=num_blocks_per_stage[3] if use_se else 0)
        self.gap = layers.GlobalAveragePooling2D()
        self.linear = layers.Dense(num_classes)

    def _make_stage(self,
                    planes: int,
                    num_blocks: int,
                    num_se_blocks: int) -> tf.keras.Sequential:
        """ Build a stage of MobileOne model.

        :param planes: Number of output channels.
        :param num_blocks: Number of blocks in this stage.
        :param num_se_blocks: Number of SE blocks in this stage.
        :return: A stage of MobileOne model.
        """
        # Get strides for all layers
        strides = [2] + [1]*(num_blocks-1)
        blocks = []
        for ix, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot "
                                 "exceed number of layers.")
            if ix >= (num_blocks - num_se_blocks):
                use_se = True

            # should check again whether to follow this one depthwise, pointwise method(since current error is here when calling get_equivalent_kernel_bias) https://github.com/notplus/MobileOne-TF2/blob/main/mobileone.py#L61
            # Depthwise conv
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=self.in_planes,
                                         kernel_size=3,
                                         stride=stride,
                                         padding='same',
                                         groups=self.in_planes,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            # Pointwise conv
            blocks.append(MobileOneBlock(in_channels=self.in_planes,
                                         out_channels=planes,
                                         kernel_size=1,
                                         stride=1,
                                         padding='valid',
                                         groups=1,
                                         inference_mode=self.inference_mode,
                                         use_se=use_se,
                                         num_conv_branches=self.num_conv_branches))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return tf.keras.Sequential(blocks)

    def call(self, x):
        """ Apply forward pass. """
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = layers.Flatten()(x)
        x = self.linear(x)
        return x


PARAMS = {
    "s0": {"width_multipliers": (0.75, 1.0, 1.0, 2.0),
           "num_conv_branches": 4},
    "s1": {"width_multipliers": (1.5, 1.5, 2.0, 2.5)},
    "s2": {"width_multipliers": (1.5, 2.0, 2.5, 4.0)},
    "s3": {"width_multipliers": (2.0, 2.5, 3.0, 4.0)},
    "s4": {"width_multipliers": (3.0, 3.5, 3.5, 4.0),
           "use_se": True},
}

def mobileone(num_classes: int = 1000, inference_mode: bool = False,
              variant: str = "s0"):
    """Get MobileOne model.

    :param num_classes: Number of classes in the dataset.
    :param inference_mode: If True, instantiates model in inference mode.
    :param variant: Which type of model to generate.
    :return: MobileOne model. """
    variant_params = PARAMS[variant]
    return MobileOne(num_classes=num_classes, inference_mode=inference_mode,
                     **variant_params)

def reparameterize_model(model, variant='s0', num_classes=1000, input_size=(224,224,3), save_path=None):
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    i = tf.random.normal((4, *input_size))
    deploy_model = mobileone(variant=variant, num_classes=num_classes, inference_mode=True)
    deploy_model(i)
    # deploy_model.build(input_shape=(None, *input_size))

    '''
    # S0 conv_branch=4
    (_layers:0~6)
    0:stem
    1~4:
        stage1(2 Block): 
            Block[0]:
                MbBlock 3*3 depth-wise #(merge into reparam_conv, and use rbr_conv[0] shape as reparam_conv shape)
                    - rbr_conv[] (conv_branch=4) # when reparm will merge 4 conv_bn into 1
                    - rbr_scale
                    - rbr_skip
                MbBlock 1*1 point-wise (error here)
                    - rbr_conv[] (conv_branch=4) 
                    - rbr_scale
                    - rbr_skip
            Block[1]:
                MbBlock 3*3 depth-wise
                    ...
                MbBlock 1*1 point-wise
                    ...

        stage2(8 Block):
            Block[0]:
            ...
            Block[7]:
        stage3(10 Block):
            ...
        stage4(1 Block):
            ...

    5:gap
    6:Dense
    '''
    for i, (module, deploy_module) in enumerate(zip(model._layers, deploy_model._layers)):
        print(f"Reparam layer i: {i}")
        if hasattr(module, 'get_equivalent_kernel_bias'):
            kernel, bias = module.get_equivalent_kernel_bias()
            deploy_module.reparam_conv.set_weights([kernel, bias])
        
        elif isinstance(module, tf.keras.Sequential):
            assert isinstance(deploy_module, tf.keras.Sequential)
            for j, (mod, deploy_mod) in enumerate(zip(module.layers, deploy_module.layers)):
                if hasattr(mod, 'get_equivalent_kernel_bias'):
                    print(f"Reparam layer j: {j}")
                    # if i==1 and j==2:
                    #     print("err her")
                    kernel, bias = mod.get_equivalent_kernel_bias()
                    deploy_mod.reparam_conv.set_weights([kernel, bias])
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        
        #Save Only the Model Weights
        weight_path = f"{save_path}/.weights.h5"
        deploy_model.save_weights(weight_path)

        #Save the whole
        # deploy_model.save(save_path)
        deploy_model.export(save_path)


    return deploy_model


    ''''''
    # Avoid editing original graph
    # model = copy.deepcopy(model)
    for i, module in enumerate(model._layers):
        if hasattr(module, 'reparameterize'):
            print(f"Reparam layer i: {i}")
            module.reparameterize()
        elif isinstance(module, tf.keras.Sequential):
            for j, mod in enumerate(module.layers):
                if hasattr(mod, 'reparameterize'):
                    print(f"Reparam layer j: {j}")
                    mod.reparameterize()
    return model
