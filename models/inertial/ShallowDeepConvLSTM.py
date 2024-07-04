# ------------------------------------------------------------------------
# Shallow DeepConvLSTM model suggested by Bock et al.
# https://doi.org/10.1145/3460421.3480419
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock@uni-siegen.de
# ------------------------------------------------------------------------

from torch import nn


class ShallowDeepConvLSTM(nn.Module):
    """
    ShallowDeepConvLSTM implementation by Bock et al. (2021).

    Args:
        channels (int): Number of input channels.
        classes (int): Number of output classes.
        window_size (int): Size of the input window.
        conv_kernels (int, optional): Number of convolutional kernels. Defaults to 64.
        conv_kernel_size (int, optional): Size of the convolutional kernels. Defaults to 5.
        lstm_units (int, optional): Number of LSTM units. Defaults to 128.
        lstm_layers (int, optional): Number of LSTM layers. Defaults to 2.
        dropout (float, optional): Dropout probability. Defaults to 0.5.
    """

    def __init__(self, channels, classes, window_size, conv_kernels=64, conv_kernel_size=5, lstm_units=128, lstm_layers=2, dropout=0.5):
        super(ShallowDeepConvLSTM, self).__init__()

        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.lstm = nn.LSTM(channels * conv_kernels, lstm_units, num_layers=lstm_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, classes)
        self.activation = nn.ReLU()
        self.final_seq_len = window_size - (conv_kernel_size - 1) * 4
        self.lstm_units = lstm_units
        self.classes = classes

    def forward(self, x):
        """
        Forward pass of the ShallowDeepConvLSTM module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, channels).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, classes).
        """
        x = x.unsqueeze(1)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, h = self.lstm(x)
        x = x.view(-1, self.lstm_units)
        x = self.dropout(x)   
        x = self.classifier(x)
        out = x.view(-1, self.final_seq_len, self.classes)
        return out[:, -1, :]
    