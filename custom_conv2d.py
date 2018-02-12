import tensorflow as tf
import math


def im2col(input_tensor, kernel_shape, strides):
    shape = tf.shape(input_tensor) # todo: decide whether it should be dynamic or not (make two versions)
    
    rows, cols = list(map(lambda x: int(x), input_tensor.get_shape()[1:3]))
    row_extent = rows - kernel_shape[0] + 1
    col_extent = cols - kernel_shape[1] + 1

    # Get indices for for the pixels forming a patch
    patch_idx = tf.range(kernel_shape[0])[:, None] * cols + tf.range(kernel_shape[1])

    # Get offset indices across height and width at which patches will be extracted
    offset_idx = tf.range(row_extent)[::strides[1], None] * cols + tf.range(col_extent)[::strides[2]]
    
    # Get dimensions of the new image
    out_rows, out_cols = list(map(lambda x: int(x), offset_idx.get_shape()))

    # Get all the actual indices
    indices = tf.reshape(offset_idx, [-1, 1]) + tf.reshape(patch_idx, [-1])
    
    # Flatten the images into one long vector
    input_flat = tf.reshape(input_tensor, [shape[0], -1, shape[3]])
    # Pick out the desired patches by indexing into tensor
    input_flat_t = tf.transpose(input_flat, perm=[1, 0, 2]) # Put the dimension to pick from first
    patches_with_chan_t = tf.gather(input_flat_t, indices) # The separate channels are still a separate dimension
    patches_with_chan = tf.transpose(patches_with_chan_t, perm=[2, 0, 1, 3]) # Put the batch dimension first again
    
    patches_shape = tf.shape(patches_with_chan)
    res = tf.reshape(patches_with_chan, [patches_shape[0], patches_shape[1], -1]) # Flatten separate channels
    return res, out_rows, out_cols

def flatten_kernel(kernel):
    return tf.reshape(kernel, [-1, tf.shape(kernel)[3]])

def pad(input_tensor, kernel_shape, strides):
    in_shape = list(map(lambda x: int(x), input_tensor.get_shape()[1:3]))
    
    out_height = math.ceil(float(in_shape[0]) / float(strides[1]))
    out_width  = math.ceil(float(in_shape[1]) / float(strides[2]))

    # Padding mimicks that in tf.nn.conv2d
    pad_along_height = max((out_height - 1) * strides[1] + kernel_shape[0] - in_shape[0], 0)
    pad_along_width = max((out_width - 1) * strides[2] + kernel_shape[1] - in_shape[1], 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    padded_tensor = tf.pad(input_tensor, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0,0]],
                         mode='CONSTANT')
    
    pad_dims = [[pad_top, pad_bottom], [pad_left, pad_right]]
    return padded_tensor, pad_dims

def conv2d(input_tensor, kernel, strides, padding='VALID'):
    kernel_shape = list(map(lambda x: int(x), kernel.get_shape()))
    if padding == 'SAME':
        input_tensor = pad(input_tensor, kernel_shape, strides)
    elif padding != 'VALID':
        raise NameError('No padding option named {}'.format(padding))
    
    patches, out_rows, out_cols = im2col(input_tensor, kernel_shape, strides)
    
    flattened_kernel = flatten_kernel(kernel)
    
    # Reshape patches as tf.matmul only works on 2D matrices (eventually try tiling kernel)
    patches_flat = tf.reshape(patches, [-1, tf.shape(patches)[2]])
    convolved_flat = tf.matmul(patches_flat, flattened_kernel)
    
    return tf.reshape(convolved_flat, [tf.shape(patches)[0], out_rows, out_cols, -1])
