import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt

def make_cell_model():
    model = tf.keras.Sequential()
    
    # perception vector will be num_channels * 3 (sobel filter x, 
    # sobel filter y, identity) for each pixel in image of H, W
    model.add(layers.Dense(128, input_shape=(HEIGHT, WIDTH, CHANNELS * 3,)))
    model.add(layers.ReLU())

    model.add(layers.Dense(CHANNELS))

    print(model.summary())
    return model

@tf.function
def perceive(x):
    identity = np.float32([0, 1, 0])
    identity = np.outer(identity, identity) # outer product of I w/ I yields I in R^3x3
    dx = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0 # sobel filter
    dy = dx.T
    kernel = tf.stack([identity, tf.cast(dx, tf.float32), tf.cast(dy, tf.float32)], -1)[:, :, None, :]
    kernel = tf.repeat(kernel, CHANNELS, 2)
    y = tf.nn.depthwise_conv2d(x, kernel, [1, 1, 1, 1], "SAME")
    return y

@tf.function
def evaluate(x, model):
    perception_image = perceive(x)
    dx = model(perception_image) * STEP_SIZE
    x += dx # update whoooo!
    alive = living_mask(x)
    return x * tf.cast(alive, tf.float32) # to turn from bools to floats

@tf.function
def train_step(x, num_iters):
    with tf.GradientTape() as g:
        for _ in tf.range(num_iters):
            x = evaluate(x, model)
        try:
            l = tf.reduce_mean(loss(x))
        except Exception as e:
            print(num_iters)
    grads = g.gradient(l, model.weights)
    grads = [g / (tf.norm(g) + 1e-8) for g in grads] # make sure we're not dividing by 0 but normalize gradients
    optimizer.apply_gradients((zip(grads, model.weights)))
    return x, l

def toAlpha(x):
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0) # has to be between 0 and 1

def toRGBA(x):
    return x[..., :4] # r, g, b, alpha

def living_mask(x):
    '''From the HEIGHT * WIDTH * CHANNELS image, get the 
       living values.
       '''
    alpha = x[:, :, 3:4]
    return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], "SAME") > 0.1 # alive is w/ alpha > 0.1

def make_seed():
#   x = np.zeros([HEIGHT, WIDTH, CHANNELS], np.float32)
  x = np.random.rand(HEIGHT, WIDTH, CHANNELS).astype(np.float32)
  x[HEIGHT // 2, WIDTH // 2, 2:] = 1.0
  return x

def load_image(file):
    img = Image.open(file)
    img.thumbnail((TARGET_SIZE, TARGET_SIZE), PIL.Image.ANTIALIAS)
    img = np.float32(img) / 255.0 # to map [0, 255] to range [0, 1]
    img[..., :3] *= img[..., 3:] # fourth layer of a PIL Image is the transparency layer
    return img

def loss(x):
    return torch.image_e
    # return tf.reduce_mean(tf.square(toRGBA(x) - TARGET), [-2, -3, -1])

def get_evaluation_iter(i):
    if i < WARMUP_ITERATIONS:
        return MIN_EVAL_ITER
    i_adjusted = (i - WARMUP_ITERATIONS) / (NUM_ITERS - WARMUP_ITERATIONS)
    return int( (MAX_EVAL_ITER - MIN_EVAL_ITER) * i_adjusted + MIN_EVAL_ITER )

CHANNELS = 16
TARGET_SIZE = 64
STEP_SIZE = 0.01
TARGET = load_image("cell.png")
HEIGHT, WIDTH = TARGET.shape[0], TARGET.shape[1]
MAX_EVAL_ITER = 32
MIN_EVAL_ITER = 1
WARMUP_ITERATIONS = 50
BATCH_SIZE = 10
NUM_ITERS = 200 # 100

model = make_cell_model()
seed = make_seed()
optimizer = tf.keras.optimizers.legacy.Adam()

times = []
losses = []
for i in range(NUM_ITERS):
    x0 = np.repeat(seed[None, ...], BATCH_SIZE, 0)
    evaluation_iter = get_evaluation_iter(i)
    x, loss = train_step(x0, 2)
    losses.append(evaluation_iter)
    times.append(i)
    print(loss)
plt.imshow(x[0, :, :, 3])
plt.show()
plt.plot(times, losses)
plt.show()
# need to fix pooling // batching implementations -- how to get sizes to agree? // multiply all things in the batch // pool