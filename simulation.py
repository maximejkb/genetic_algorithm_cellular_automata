import cv2 as cv
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import pickle


SIM_CANVAS_SHAPE = (60, 60) # (50, 50)# 
DISPLAY_CANVAS_SHAPE = (960, 540) # 1080p aspect ratio
KERNEL_SHAPE = (3, 3)
N_STEPS = 600
SAVE_STEPS = 1
SCALE_FACTOR = 2 # scale by 2 to get to 1080p
N_PLOT_ROWS, N_PLOT_COLS, N_PLOT_SAMPLES = 3, 3, 9
FPS = 15


def binarize(img):
    '''Applies adaptive thresholding to img and returns the
       resulting image.
       '''
    # Read image as grayscale (mode 0).
    im = cv.imread(img, 0)

    # Block size must be odd.
    BLOCK_SIZE = 201
    # Bias subtracted from averaged value before local threshold comparison.
    BIAS = 4

    thresholded =  cv.adaptiveThreshold(im, 255,
                    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv.THRESH_BINARY,
                    BLOCK_SIZE, BIAS)

    # plt.imshow(thresholded, "gray")
    # plt.show()

    return thresholded


def readKernel(file):
    '''Kernels have white where squares shouldn't be counted and 
       black where they should be. We need to reverse that
       '''
    im = cv.imread(file, 0)
    return 1 - np.round(im / 255)


def blur(k, im, kernel_shape):
    '''Blur the image k times.
       '''
    if k == 1:
        return cv.blur(im, kernel_shape)
    return blur(k - 1, cv.blur(im, kernel_shape), kernel_shape)


def conway():
    kernel = readKernel("conwaykernel.jpg")
    # initial = np.zeros(SIM_CANVAS_SHAPE)
    # initial[1][0] = initial[2][1] = initial[2][2] = initial[1][2] = initial[0][2] = 1
    initial = randomInitial(0.5) # centerSeedInitial(10, 10) #
    life_rule = lambda last, k: (((k == 2) | (k == 3)) & (last == 1)) | ((k == 3) & (last == 0))
    update_rule = lambda last, k: life_rule(last, k)

    # run it!
    runCellularAutomata(np.repeat(np.expand_dims(initial, 0), 2, 0), np.repeat(np.expand_dims(kernel, 0), 2, 0), [update_rule, update_rule], N_STEPS, SAVE_STEPS, 1, True)


def roundCellsGrowing(initial):
    # initialize w/ seed arrangement
    #     # initial = randomInitial(0.6)

    # define convolution kernel
    kernel0 = readKernel("kernel0.jpg")
    kernel1 = readKernel("kernel1.jpg")
    kernel2 = readKernel("kernel2.jpg")
    kernel3 = readKernel("kernel3.jpg")

    # define update rules given the last cell states and the output from kernel
    death_rule = lambda sum_0, sum_1, sum_2, sum_3: (sum_0 <= 17) | ((sum_2 >= 9) & (sum_2 <= 21)) | ((sum_3 >= 78) & (sum_3 <= 89)) | (sum_3 > 108) # mark cells for death with 1
    life_rule = lambda sum_0, sum_1, sum_2, sum_3:  (sum_1 >= 6) # mark cells for life with 1
    update_rule = lambda last, k0, k1, k2, k3: life_rule(k0, k1, k2, k3) & ~death_rule(k0, k1, k2, k3)

    # run it!
    # np.stack([kernel0, kernel1, kernel2, kernel3])
    runCellularAutomata(initial, [kernel0, kernel1, kernel2, kernel3], update_rule, N_STEPS, SAVE_STEPS, 0, display=True)


def randomInitial(p):
    '''Return an initial board where each cell is alive with probability p.
       '''
    initial = np.random.rand(*SIM_CANVAS_SHAPE) 
    return (initial < p).astype(float)


def centerSeedInitial(seedWidth, seedHeight):
    '''Return an initial board w/ a seed in center.
       '''
    initial = np.zeros(SIM_CANVAS_SHAPE)
    xCenter, yCenter = SIM_CANVAS_SHAPE[1] / 2, SIM_CANVAS_SHAPE[0] / 2 - 1
    left, right = int(xCenter - seedWidth / 2), int(xCenter + seedWidth / 2)
    top, bottom = int(yCenter - seedHeight / 2), int(yCenter + seedHeight / 2)
    initial[top: bottom, left:right] = 1
    return initial


def writeVideo(frames):
    h, w = DISPLAY_CANVAS_SHAPE[1] * SCALE_FACTOR, DISPLAY_CANVAS_SHAPE[0] * SCALE_FACTOR # reversed b/c numpy and openCV used inverse conventions 
    videoFile = "{}-steps_every-{}-steps_{}-fps_{}".format(N_STEPS, SAVE_STEPS, FPS, hash(str(initial)))
    out = cv.VideoWriter("videos/{}.mp4".format(videoFile), cv.VideoWriter_fourcc(*'mp4v') , FPS, (w, h)) # flip w and h again bc going to transpose
    n = len(frames)
    bar_length = 20
    for i, f in enumerate(frames):
        t = np.transpose(f, (1, 0, 2))
        out.write(t.astype("uint8"))
        done = i / n
        bar_done = int(bar_length * done)
        bar = "[" + "-" * int(bar_done)  + ">" + " " * (bar_length - bar_done) + "]"
        print("writing frame {0} / {1}; {2:.2f}% complete; {3}".format(i, n, 100 * done, bar), end="\r", flush=True)
    out.release()


binary_cmap = ListedColormap(['black', 'white'])
def display_frame(current_state, idxs, generation_number, frame_number, fullscreen_first=False):
    if not fullscreen_first:
        fig, ax = plt.subplots(N_PLOT_ROWS, N_PLOT_COLS, figsize=(DISPLAY_CANVAS_SHAPE[1]/96, DISPLAY_CANVAS_SHAPE[0]/96), dpi=96, frameon=False)
        for i in range(N_PLOT_SAMPLES):
            col = i % N_PLOT_COLS
            row = i // N_PLOT_COLS
            axis = ax[row, col]
            state = current_state[0][idxs[i]]
            axis.imshow( state, vmin=0, vmax=1, cmap=binary_cmap, interpolation=None)
            axis.axis("off")
        plt.tight_layout()
    else:
        state = current_state[0][idxs[0]]
        fig = plt.figure(figsize=(DISPLAY_CANVAS_SHAPE[1]/96, DISPLAY_CANVAS_SHAPE[0]/96), dpi=96, frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(state, vmin=0, vmax=1, cmap=binary_cmap, interpolation=None)
        # plt.imshow(state, vmin=0, vmax=1, cmap='jet')
        plt.axis("off")
    plt.savefig(f"ca-images/{generation_number}_{frame_number}.png", transparent=True)
    plt.close() 


def runCellularAutomata(initials, kernels, rules, num_iters, save_num_iters, generation, display=True, plot_idxs=None, fullscreen_display=False):
    state = torch.Tensor(initials).unsqueeze(0).detach()
    kernels = torch.Tensor(kernels).unsqueeze(1).detach()
    C_in, B, H, W = state.size()
    if plot_idxs is None:
        plot_idxs = np.random.choice(B, N_PLOT_SAMPLES)   
    for i in range(num_iters):
        if display and i % save_num_iters == 0:
            display_frame(state, plot_idxs, generation, i, fullscreen_first=fullscreen_display)
        filters = torch.nn.functional.conv2d(state, kernels, groups=B, padding="same")
        state = torch.stack([r(s, f) for r, s, f in zip(rules, state.transpose(1, 0), filters.transpose(1, 0))]).transpose(0, 1).float()
    return state #.squeeze(0).numpy()
    

if __name__ == "__main__":
    from genetic_algorithm import make_kernels, make_rules, top_k_ranked_organism_idxs, score, POPULATION_SIZE
    SIM_CANVAS_SHAPE = DISPLAY_CANVAS_SHAPE

    seed = centerSeedInitial(7, 7) # randomInitial(0.5) #
    x0 = np.repeat(seed[None, ...], POPULATION_SIZE, 0)
    with open("population.pkl", "rb") as f:
        population = pickle.load(f)
    kernel, update_rule = make_kernels(population), make_rules(population)
    final = runCellularAutomata(
        x0, 
        kernel, 
        update_rule, 
        N_STEPS, 
        -1, 
        0, 
        display=True,
        plot_idxs=None,
        fullscreen_display=True
    )