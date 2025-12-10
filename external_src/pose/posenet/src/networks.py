import torch
from . import net_utils


'''
Encoders
'''
class ResNetEncoder(torch.nn.Module):
    '''
    ResNet encoder with skip connections

    Arg(s):
        n_layer : int
            architecture type based on layers: 18, 34, 50
        input_channels : int
            number of channels in input data
        n_filters : list
            number of filters to use for each block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 n_layer,
                 input_channels=3,
                 n_filters=[32, 64, 128, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(ResNetEncoder, self).__init__()

        if n_layer == 18:
            n_blocks = [2, 2, 2, 2]
            resnet_block = net_utils.ResNetBlock
        elif n_layer == 34:
            n_blocks = [3, 4, 6, 3]
            resnet_block = net_utils.ResNetBlock
        else:
            raise ValueError('Only supports 18, 34 layer architecture')

        for n in range(len(n_filters) - len(n_blocks) - 1):
            n_blocks = n_blocks + [n_blocks[-1]]

        network_depth = len(n_filters)

        assert network_depth < 8, 'Does not support network depth of 8 or more'
        assert network_depth == len(n_blocks) + 1

        # Keep track on current block
        block_idx = 0
        filter_idx = 0

        activation_func = net_utils.activation_func(activation_func)

        in_channels, out_channels = [input_channels, n_filters[filter_idx]]

        # Resolution 1/1 -> 1/2
        self.conv1 = net_utils.Conv2d(
            in_channels,
            out_channels,
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        # Resolution 1/2 -> 1/4
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1)

        filter_idx = filter_idx + 1

        blocks2 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):

            if n == 0:
                pass
            else:
                in_channels = out_channels

            stride = 1

            block = resnet_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks2.append(block)

        self.blocks2 = torch.nn.Sequential(*blocks2)

        # Resolution 1/4 -> 1/8
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        blocks3 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):

            if n == 0:
                stride = 2
            else:
                in_channels = out_channels
                stride = 1

            block = resnet_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks3.append(block)

        self.blocks3 = torch.nn.Sequential(*blocks3)

        # Resolution 1/8 -> 1/16
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        blocks4 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):

            if n == 0:
                stride = 2
            else:
                in_channels = out_channels
                stride = 1

            block = resnet_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks4.append(block)

        self.blocks4 = torch.nn.Sequential(*blocks4)

        # Resolution 1/16 -> 1/32
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        blocks5 = []
        in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
        for n in range(n_blocks[block_idx]):

            if n == 0:
                stride = 2
            else:
                in_channels = out_channels
                stride = 1

            block = resnet_block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                weight_initializer=weight_initializer,
                activation_func=activation_func,
                use_batch_norm=use_batch_norm)

            blocks5.append(block)

        self.blocks5 = torch.nn.Sequential(*blocks5)

        # Resolution 1/32 -> 1/64
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            blocks6 = []
            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
            for n in range(n_blocks[block_idx]):

                if n == 0:
                    stride = 2
                else:
                    in_channels = out_channels
                    stride = 1

                block = resnet_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)

                blocks6.append(block)

            self.blocks6 = torch.nn.Sequential(*blocks6)
        else:
            self.blocks6 = None

        # Resolution 1/64 -> 1/128
        block_idx = block_idx + 1
        filter_idx = filter_idx + 1

        if filter_idx < len(n_filters):

            blocks7 = []
            in_channels, out_channels = [n_filters[filter_idx-1], n_filters[filter_idx]]
            for n in range(n_blocks[block_idx]):

                if n == 0:
                    stride = 2
                else:
                    in_channels = out_channels
                    stride = 1

                block = resnet_block(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=stride,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)

                blocks7.append(block)

            self.blocks7 = torch.nn.Sequential(*blocks7)
        else:
            self.blocks7 = None

    def forward(self, x):
        '''
        Forward input x through the ResNet model

        Arg(s):
            x : torch.Tensor
        Returns:
            torch.Tensor[float32] : latent vector
            list[torch.Tensor[float32]] : skip connections
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        max_pool = self.max_pool(layers[-1])
        layers.append(self.blocks2(max_pool))

        # Resolution 1/4 -> 1/8
        layers.append(self.blocks3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.blocks4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.blocks5(layers[-1]))

        # Resolution 1/32 -> 1/64
        if self.blocks6 is not None:
            layers.append(self.blocks6(layers[-1]))

        # Resolution 1/64 -> 1/128
        if self.blocks7 is not None:
            layers.append(self.blocks7(layers[-1]))

        return layers[-1], layers[1:-1]


class PoseEncoder(torch.nn.Module):
    '''
    Pose network encoder

    Arg(s):
        input_channels : int
            number of channels in input data
        n_filters : list[int]
            number of filters to use for each convolution
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 input_channels=6,
                 n_filters=[16, 32, 64, 128, 256, 256, 256],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(PoseEncoder, self).__init__()

        activation_func = net_utils.activation_func(activation_func)

        self.conv1 = net_utils.Conv2d(
            input_channels,
            n_filters[0],
            kernel_size=7,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv2 = net_utils.Conv2d(
            n_filters[0],
            n_filters[1],
            kernel_size=5,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv3 = net_utils.Conv2d(
            n_filters[1],
            n_filters[2],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv4 = net_utils.Conv2d(
            n_filters[2],
            n_filters[3],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv5 = net_utils.Conv2d(
            n_filters[3],
            n_filters[4],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv6 = net_utils.Conv2d(
            n_filters[4],
            n_filters[5],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

        self.conv7 = net_utils.Conv2d(
            n_filters[5],
            n_filters[6],
            kernel_size=3,
            stride=2,
            weight_initializer=weight_initializer,
            activation_func=activation_func,
            use_batch_norm=use_batch_norm)

    def forward(self, x):
        '''
        Forward input x through the ResNet model

        Arg(s):
            x : torch.Tensor[float32]
                N x 6 x H x W tensor
        Returns:
            torch.Tensor[float32] : latent vector
            list[torch.Tensor[float32]] : skip connections
        '''

        layers = [x]

        # Resolution 1/1 -> 1/2
        layers.append(self.conv1(layers[-1]))

        # Resolution 1/2 -> 1/4
        layers.append(self.conv2(layers[-1]))

        # Resolution 1/4 -> 1/8
        layers.append(self.conv3(layers[-1]))

        # Resolution 1/8 -> 1/16
        layers.append(self.conv4(layers[-1]))

        # Resolution 1/16 -> 1/32
        layers.append(self.conv5(layers[-1]))

        # Resolution 1/32 -> 1/64
        layers.append(self.conv6(layers[-1]))

        # Resolution 1/64 -> 1/128
        layers.append(self.conv7(layers[-1]))

        return layers[-1], None


'''
Decoder Architectures
'''
class PoseDecoder(torch.nn.Module):
    '''
    Pose Decoder 6 DOF

    Arg(s):
        rotation_parameterization : str
            axis
        input_channels : int
            number of channels in input latent vector
        n_filters : int list
            number of filters to use at each decoder block
        weight_initializer : str
            kaiming_normal, kaiming_uniform, xavier_normal, xavier_uniform
        activation_func : func
            activation function after convolution
        use_batch_norm : bool
            if set, then applied batch normalization
    '''

    def __init__(self,
                 rotation_parameterization,
                 input_channels=256,
                 n_filters=[],
                 weight_initializer='kaiming_uniform',
                 activation_func='leaky_relu',
                 use_batch_norm=False):
        super(PoseDecoder, self).__init__()

        self.rotation_parameterization = rotation_parameterization

        activation_func = net_utils.activation_func(activation_func)

        if len(n_filters) > 0:
            layers = []
            in_channels = input_channels

            for out_channels in n_filters:
                conv = net_utils.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=2,
                    weight_initializer=weight_initializer,
                    activation_func=activation_func,
                    use_batch_norm=use_batch_norm)
                layers.append(conv)
                in_channels = out_channels

            conv = net_utils.Conv2d(
                in_channels=in_channels,
                out_channels=6,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=None,
                use_batch_norm=False)
            layers.append(conv)

            self.conv = torch.nn.Sequential(*layers)
        else:
            self.conv = net_utils.Conv2d(
                in_channels=input_channels,
                out_channels=6,
                kernel_size=1,
                stride=1,
                weight_initializer=weight_initializer,
                activation_func=None,
                use_batch_norm=False)

    def forward(self, x):
        '''
        Forward latent vector x through decoder network

        Arg(s):
            x : torch.Tensor[float32]
                latent vector
        Returns:
            torch.Tensor[float32] : N x 6 vector
        '''

        conv_output = self.conv(x)
        pose_mean = torch.mean(conv_output, [2, 3])
        dof = 0.01 * pose_mean
        posemat = net_utils.pose_matrix(
            dof,
            rotation_parameterization=self.rotation_parameterization)

        return posemat
