import tensorflow as tf
from tensorflow.keras import layers, models # type: ignore
# import torch.nn as nn
# c = layers.Conv2D(
#     filters = 6,
#     kernel_size = 5,
#     strides = 1,
#     padding = 'same'
# )
# i = tf.random.normal([1,5,5,3])
# print(c(i).shape)

'''
in_channels = 6
out_channels = 3
stride = 1
rbr_skip = nn.BatchNorm2d(num_features=in_channels) \
    if out_channels == in_channels and stride == 1 else None
print(rbr_skip)
'''

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
        super().__init__()
        self.conv = self._conv_bn(3, 'valid')

    def reparameterize(self):
        ...

    def _conv_bn(
            self,
            kernel_size: int, 
            padding: str
        ):
        module = tf.keras.Sequential()
        module.add(
            layers.Conv2D(
                filters = 6,
                kernel_size = kernel_size,
                strides = 1,
                padding = padding,
                use_bias = False,
                name="conv"
            )
        )
        module.add(
            layers.BatchNormalization(axis=-1, name="bn")
        )
        return (module)
    
    def call(self, x):
        return self.conv(x)

class MobileOneModel(layers.Layer):
    def __init__(
            self
        ):
        super().__init__()
        self.mb = MobileOneBlock(3, 6, 3)
        self.mb2 = self._make_stage()
    
    def _make_stage(self) -> tf.keras.Sequential:
        blocks = []
        blocks.append(MobileOneBlock(4,7,4))
        blocks.append(MobileOneBlock(5,8,5))
        return tf.keras.Sequential(blocks)

    def call(self, x):
        return self.mb(x)
        

# c = _conv_bn(3, 'valid')
# print(c(i).shape)
# print(c)

# mb = MobileOneBlock(3, 6, 3)
# mm = MobileOneModel(mb)

i = tf.random.normal([10, 32, 32, 3])
mm = MobileOneModel()
print(mm)

'''
# i = tf.random.normal([10, 32, 32, 3])
i = tf.random.normal([1, 3, 3, 1])
pd = tf.keras.layers.ZeroPadding2D(padding = ((1,1),(1,1)))
print(pd(i).shape)
print(pd(i))
print(tf.keras.layers.ZeroPadding2D(padding = ((1,1),(1,1)))(i))
'''