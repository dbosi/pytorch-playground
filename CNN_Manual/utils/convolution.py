import torch

# Manual Implementation Too Slow
def conv_2d(x: torch.Tensor, kernel: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    channel_out, channel_in, k_rows, k_cols = kernel.shape
    batch_n, channel_n, h_in, w_in = x.shape

    if channel_in != channel_n:
        raise ValueError("Kernel Channels and Input Channels numbers do not match!")

    o = torch.empty(batch_n, channel_out, h_in - k_rows + 1, w_in - k_cols + 1)

    for img_o in range(o.size(0)):
        for channel_o in range(o.size(1)):
            for h_pixel in range(o.size(2)):
                for w_pixel in range(o.size(3)):
                    patch_sum = 0

                    for channel_i in range(channel_n):
                        input_patch = x[img_o, channel_i, h_pixel:h_pixel+k_rows, w_pixel:w_pixel+k_cols]

                        kernel_patch = kernel[channel_o, channel_i]

                        patch_sum += (input_patch * kernel_patch).sum()

                    o[img_o, channel_o, h_pixel, w_pixel] = patch_sum + bias[channel_o]

    return o

# Manual Implementation Too Slow
def max_pool_2d(x: torch.Tensor, pool_size: int, stride: int) -> torch.Tensor:
    batch_n, channels_n, h, w = x.shape

    o = torch.empty(batch_n, channels_n, (h - pool_size) // stride + 1, (w - pool_size) // stride + 1)

    for img in range(o.size(0)):
        for channel in range(o.size(1)):
            for h_pixel in range(o.size(2)):
                for w_pixel in range(o.size(3)):
                    h_start = h_pixel * stride
                    w_start = w_pixel * stride

                    patch = x[img, channel, h_start:h_start+pool_size, w_start:w_start+pool_size]

                    o[img, channel, h_pixel, w_pixel] = patch.max()

    return o