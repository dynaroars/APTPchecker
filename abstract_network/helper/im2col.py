import math
import numpy as np
import torch
import torch.nn as nn


def im2col(
    data_im: np.ndarray,
    kernel_h: int,
    kernel_w: int,
    pad_h: int,
    pad_w: int,
    stride_h: int,
    stride_w: int,
    dilation_h: int,
    dilation_w: int,
) -> np.ndarray:
    """
    Convert image data to column format for convolution operations.

    Args:
        data_im: Input image data
        channels: Number of input channels
        height: Input height
        width: Input width
        output_height: Output height
        output_width: Output width
        kernel_h: Kernel height
        kernel_w: Kernel width
        pad_h: Padding height
        pad_w: Padding width
        stride_h: Stride height
        stride_w: Stride width
        dilation_h: Dilation height
        dilation_w: Dilation width

    Returns:
        Columnized data
    """
    assert len(data_im.shape) == 3
    channels, height, width = data_im.shape
    data_im = data_im.flatten()
    height_col = math.floor((height + 2 * pad_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1)
    width_col = math.floor((width + 2 * pad_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1)
    channels_col = channels * kernel_h * kernel_w

    data_col = np.zeros((channels_col * height_col * width_col,), dtype=data_im.dtype)

    for c_col in range(channels_col):
        w_offset = c_col % kernel_w
        h_offset = (c_col // kernel_w) % kernel_h
        c_im = c_col // kernel_h // kernel_w

        for h_col in range(height_col):
            h_im = h_col * stride_h - pad_h + h_offset * dilation_h

            for w_col in range(width_col):
                w_im = w_col * stride_w - pad_w + w_offset * dilation_w

                if 0 <= h_im < height and 0 <= w_im < width:
                    data_col[(c_col * height_col + h_col) * width_col + w_col] = data_im[
                        (c_im * height + h_im) * width + w_im
                    ]

    return data_col.reshape(channels_col, height_col * width_col)


if __name__ == "__main__":
    data_im = np.random.randn(1, 5, 3, 4)
    padding = (0, 0)
    stride = (1, 1)
    dilation = (1, 1)
    kernel_size = (2, 3)
    data_col = im2col(
        data_im.flatten(),
        5,
        3,
        4,
        kernel_size[0],
        kernel_size[1],
        padding[0],
        padding[1],
        stride[0],
        stride[1],
        dilation[0],
        dilation[1],
    )
    print(data_col)

    print(
        nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)(
            torch.from_numpy(data_im)
        )
    )
