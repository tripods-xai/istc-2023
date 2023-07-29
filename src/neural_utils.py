import torch
import torch.nn.functional as F

# utility for Same Shape CNN 1D
class SameShapeConv1d(torch.nn.Module):
    def __init__(
        self,
        num_layer,
        in_channels,
        out_channels,
        kernel_size,
        activation="elu",
        no_act=False,
        first_pad=False,
        front_pad=False,
        pad_value=0.0,
    ):
        super(SameShapeConv1d, self).__init__()

        self.cnns = torch.nn.ModuleList()
        self.num_layer = num_layer
        self.no_act = no_act
        self.kernel_size = kernel_size
        self.first_pad = first_pad
        self.front_pad = front_pad
        self.first_padding_layer = (
            torch.nn.ConstantPad1d(
                self.make_pad_tuple(self._out_shape_reduced_count()), value=pad_value
            )
            if self.first_pad
            else None
        )
        self.padding_layers = torch.nn.ModuleList()
        for idx in range(num_layer):
            if idx == 0:
                self.cnns.append(
                    torch.nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        groups=1,
                        bias=True,
                    )
                )
            else:
                self.cnns.append(
                    torch.nn.Conv1d(
                        in_channels=out_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        groups=1,
                        bias=True,
                    )
                )
            if not self.first_pad:
                self.padding_layers.append(
                    torch.nn.ConstantPad1d(
                        self.make_pad_tuple(kernel_size - 1), value=pad_value
                    )
                )

        if activation == "elu":
            self.activation = F.elu
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "selu":
            self.activation = F.selu
        elif activation == "prelu":
            self.activation = F.prelu
        else:
            self.activation = F.elu

    def forward(self, inputs, prepadded=False):
        inputs = torch.transpose(inputs, 1, 2)
        x = inputs
        if (self.first_padding_layer is not None) and not prepadded:
            x = self.first_padding_layer(x)
        for idx in range(self.num_layer):
            if idx != 0 or (not prepadded):
                if not self.first_pad:
                    x = self.padding_layers[idx](x)
            if self.no_act:
                x = self.cnns[idx](x)
            else:
                x = self.activation(self.cnns[idx](x))

        outputs = torch.transpose(x, 1, 2)
        return outputs

    def make_pad_tuple(self, pad):
        if self.front_pad:
            return (pad, 0)
        else:
            return (pad // 2, pad - pad // 2)

    def _out_shape_reduced_count(self):
        return (self.kernel_size - 1) * self.num_layer
