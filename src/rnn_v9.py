import custom_loss
import tensorflow as tf
from tensorflow.keras import layers

# model fitting hyperparameters
epochs = 200
patience = 10
batch_size = 256

# compile parameters
optimizer = tf.keras.optimizers.Adam
opt_hyperparameters = {
    "learning_rate": 0.001,
    "beta_1": 0.9,
    "beta_2": 0.999,
    "epsilon": 1e-7,
}
loss = {"shape": "CosineSimilarity", "sum": "msle"}
metrics = {"shape": custom_loss.corr}

# model architecture hyperparameters
c1 = {"filters": 64, "kernel_size": 8}
c2 = {"filters": 128, "kernel_size": 4}
num_blocks = [2, 2, 2, 2]
num_filters = [64, 128, 256, 512]
rblock_kernel = 4
dropout = 0.3
output_kernel = 9


def rblock(x, filters, change_filters, kernel_size):
    """"""
    # Project into correct filter size
    if change_filters:
        x = layers.Conv1D(kernel_size=1, filters=filters)(x)
    # 1st convolutional layer
    residual = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(
        x
    )
    residual = layers.BatchNormalization()(residual)
    residual = layers.Activation("relu")(residual)
    # 2nd convolutional layer
    residual = layers.Conv1D(filters=filters, kernel_size=kernel_size, padding="same")(
        residual
    )
    residual = layers.BatchNormalization()(residual)
    residual = layers.Activation("relu")(residual)
    # Add skip connection
    out = layers.Add()([x, residual])
    out = layers.BatchNormalization()(out)
    out = layers.Activation("relu")(out)
    return out


def construct_nn(input_length, output_length):
    """Returns a neural network with given input and output lengths."""
    # input
    X = layers.Input(shape=(input_length, 4 + 1))
    y = layers.BatchNormalization()(X)
    # 1st convolutional layer
    y = layers.Conv1D(filters=c1["filters"], kernel_size=c1["kernel_size"])(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("elu")(y)
    y = layers.MaxPooling1D(pool_size=(2))(y)
    # 2nd convolutional layer
    y = layers.Conv1D(filters=c2["filters"], kernel_size=c2["kernel_size"])(y)
    y = layers.BatchNormalization()(y)
    y = layers.Activation("relu")(y)
    y = layers.MaxPooling1D(pool_size=(2))(y)
    # residual blocks
    for i in range(len(num_blocks)):
        for j in range(num_blocks[i]):
            bool1 = j == 0
            if i == 0:
                bool2 = True
            else:
                bool2 = num_filters[i] != num_filters[i - 1]
            y = rblock(
                y,
                change_filters=(bool1 and bool2),
                filters=num_filters[i],
                kernel_size=rblock_kernel,
            )
        y = layers.MaxPooling1D(pool_size=(2))(y)
    # shape / probability distribution head
    p_head = layers.Flatten()(y)
    p_head = layers.Dense(output_length)(p_head)
    p_head = layers.BatchNormalization()(p_head)
    p_head = layers.Activation("relu")(p_head)
    p_head = layers.Dropout(dropout, name="shape")(p_head)
    # sum head
    s_head = layers.GlobalAvgPool1D()(y)
    s_head = layers.Dense(1)(s_head)
    s_head = layers.BatchNormalization()(s_head)
    s_head = layers.Activation("relu")(s_head)
    s_head = layers.Dropout(dropout, name="sum")(s_head)
    # output model
    return tf.keras.models.Model(inputs=X, outputs=[p_head, s_head])
