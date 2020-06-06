from PIL import Image
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import dill
import os

CHANNELS = 3
COLOR_RANGE = 256
FILTER_THRESHOLD = 0.01
ENABLE_LOW_FILTER = False
PRESET_BIAS_ERROR_WITH_FILTER = 1
LOSS_FUNC_KL_DIV = True
KL_EPSILON = 0.0001

def kl_div(source_density, target_density):
    if source_density == 0:
        if target_density > KL_EPSILON:
            # add a little epsilon to make this defined
            source_density = KL_EPSILON
        else:
            return 0

    if target_density == 0:
        return 0
    
    return target_density * math.log(target_density / source_density)

def se(source_density, target_density):
    return (source_density - target_density) ** 2

def loss_func(source_density, target_density):
    if LOSS_FUNC_KL_DIV:
        return kl_div(source_density, target_density)
    else:
        return se(source_density, target_density)
        

def get_histr(img, i):
    histr = [0] * COLOR_RANGE
    total_size = 0
    if not isinstance(img, str):
        img_h, img_w = img.shape[:2]
        total_size = img_h * img_w
        for h in range(img_h):
            for w in range(img_w):
                histr[img[h][w][i]] += 1
    else:
        images = os.listdir(img)
        for image in images:
            img_data = np.array(Image.open(img + image))
            img_h, img_w = img_data.shape[:2]
            total_size += img_h * img_w
            for h in range(img_h):
                for w in range(img_w):
                    histr[img_data[h][w][i]] += 1
    return [x / float(total_size) * 100 for x in histr]

def comparison_plot(target_img, source_img, mode='RGB'):
    target_histr = []
    source_histr = []

    if mode == 'CMYK':
        color_target = ('cyan', 'magenta', 'yellow', 'grey')
        color_source = ('darkcyan', 'darkmagenta', 'gold', 'black')
    else:
        color_target = ('salmon', 'limegreen','deepskyblue')
        color_source = ('maroon', 'seagreen', 'steelblue')
    f, axarr = plt.subplots(1, CHANNELS, sharey=True)
    f.set_figheight(6)
    f.set_figwidth(18)
    for i, col in enumerate(color_target):
        target_histr.append(get_histr(target_img, i))
        axarr[i].plot(target_histr[i], color=col, label='$D_{KL}$')
        axarr[i].fill_between(np.arange(0, COLOR_RANGE), target_histr[i], color=col, alpha=0.2)

    for i, col in enumerate(color_source):
        source_histr.append(get_histr(source_img, i))
        loss = sum([loss_func(x, y) for x, y in zip(source_histr[i], target_histr[i])]) / len(target_histr[i])
        if LOSS_FUNC_KL_DIV:
            axarr[i].set_title('$D_{KL}$: ' + str(loss))
        else:
            axarr[i].set_title('MSE: ' + str(loss))
        axarr[i].plot(source_histr[i], color=col, label='MSE')
        axarr[i].fill_between(np.arange(0, COLOR_RANGE), source_histr[i], color=col, alpha=0.2)

        axarr[i].set_yscale('log')
        axarr[i].set_xlim([0, COLOR_RANGE])
        axarr[i].legend()
    
    if mode == 'CMYK':
        axarr[0].set_xlabel('Cyan')
        axarr[1].set_xlabel('Magenta')
        axarr[2].set_xlabel('Yellow')
        axarr[3].set_xlabel('Black')
    else:
        axarr[0].set_xlabel('Red')
        axarr[1].set_xlabel('Green')
        axarr[2].set_xlabel('Blue')

    print('Showing histogram comparison by channel...')
    plt.show()

def optimize(source_histr, target_histr, name='default'):
    '''
    Args:
        source_histr (list), target_histr (list)
    Returns:
        a list of offset values fit source histogram to target histogram
    '''
    HISTR_SIZE = 256
    error_map = [[0 for i in range(HISTR_SIZE)] for j in range(HISTR_SIZE)]
    cur_density_map = [[0 for i in range(HISTR_SIZE)] for j in range(HISTR_SIZE)]
    prev_node_map = [[None for i in range(HISTR_SIZE)] for j in range(HISTR_SIZE)]
    for source_value, source_density in enumerate(source_histr):
        print('histogram fitting:', source_value, '/', HISTR_SIZE - 1, end='\r')
        for target_value, target_density in enumerate(target_histr):
            if source_value == 0:
                '''initialize first row'''
                if ENABLE_LOW_FILTER:
                    if target_value > 0:
                        error += PRESET_BIAS_ERROR_WITH_FILTER

                error = loss_func(source_density, target_density)
                for beginning_value in range(target_value):
                    error += loss_func(0, target_histr[beginning_value])
                if ENABLE_LOW_FILTER:
                    if target_value == 0:
                        error = 0
                prev_node = None
                density = source_density
            else:
                last_min_error, last_target = min((val, idx) for (idx, val) in enumerate(error_map[source_value - 1]))
                if source_density < FILTER_THRESHOLD and ENABLE_LOW_FILTER and target_value == last_target + 1:
                    error = last_min_error
                    prev_node = last_target
                    density = source_density
                else:
                    error = math.inf
                    for prev_target_value in range(target_value + 1):
                        if prev_target_value == target_value:
                            '''rollback operation'''
                            new_error = loss_func(source_density + cur_density_map[source_value - 1][target_value], target_density)
                            new_error += error_map[source_value - 1][target_value] 
                            new_error -= loss_func(cur_density_map[source_value - 1][target_value], target_density)
                            new_density = source_density + cur_density_map[source_value - 1][target_value]
                        else:
                            '''cur_density_map[source_value - 1][prev_target_value] must be 0, no need to rollback'''
                            new_error = loss_func(source_density, target_density)
                            new_error += error_map[source_value - 1][prev_target_value]
                            for intermedite_value in range(target_value)[prev_target_value + 1:]:
                                new_error += loss_func(0, target_histr[intermedite_value])
                            new_density = source_density

                        if new_error < error:
                            prev_node = prev_target_value
                            if ENABLE_LOW_FILTER and source_density < FILTER_THRESHOLD:
                                error = new_error + PRESET_BIAS_ERROR_WITH_FILTER
                            else:
                                error = new_error
                            density = new_density

            error_map[source_value][target_value] = error
            cur_density_map[source_value][target_value] = density
            prev_node_map[source_value][target_value] = prev_node

    with open('error_map.' + name + '.list.dill', 'wb') as f:
        dill.dump(error_map, f)
    
    with open('prev_node_map.' + name + '.list.dill', 'wb') as f:
        dill.dump(prev_node_map, f)

    '''need to prepend tail error for the last row'''
    for target_value in range(HISTR_SIZE):
        for tail_value in range(HISTR_SIZE)[target_value + 1:]:
            error_map[HISTR_SIZE - 1][target_value] += loss_func(0, target_histr[tail_value])

    min_error, target_value = min((val, idx) for (idx, val) in enumerate(error_map[HISTR_SIZE - 1]))
    node_list = [target_value]
    for source_value in reversed([i for i in range(HISTR_SIZE)]):
        if source_value == 0:
            print('histogram fitting complete       ')
            return node_list
        target_value = prev_node_map[source_value][target_value]
        node_list.insert(0, target_value)

def get_model(source_img, target_img, mode='RGB'):
    if mode == 'CMYK':
        cyan_f   = optimize(get_histr(source_img, 0), get_histr(target_img, 0), 'cyan')
        magen_f  = optimize(get_histr(source_img, 1), get_histr(target_img, 1), 'magenta')
        yellow_f = optimize(get_histr(source_img, 2), get_histr(target_img, 2), 'yellow')
        black_f  = optimize(get_histr(source_img, 3), get_histr(target_img, 3), 'black')

        model = [cyan_f, magen_f, yellow_f, black_f]
    else:
        red_f  = optimize(get_histr(source_img, 0), get_histr(target_img, 0))
        green_f = optimize(get_histr(source_img, 1), get_histr(target_img, 1))
        blue_f   = optimize(get_histr(source_img, 2), get_histr(target_img, 2))

        model = [red_f, green_f, blue_f]

    with open('trans_model.list.dill', 'wb') as f:
        dill.dump(model, f)

    return model

if __name__ == "__main__":
    mode = 'CMYK'
    LOAD_EXISTING_MODEL = True
    COMPARISON = False
    if mode == 'CMYK':
        CHANNELS = 4
        COLOR_RANGE = 256

    #target_sample_img = '{target_folder_name}\\'
    #source_sample_img = '{source_folder_name}\\'

    #target_sample_img = np.array(Image.open('{target_file_name}'))
    #source_sample_img = np.array(Image.open('{source_file_name}'))

    if COMPARISON:
        comparison_plot(target_sample_img, source_sample_img, mode)
    else:
        if LOAD_EXISTING_MODEL:
            with open('{model_name}', 'rb') as f:
                model = dill.load(f)
        else:
            model = get_model(source_sample_img, target_sample_img, mode)

        print(model)

        source_img = np.array(Image.open('{source_file_name}'))
        source_h, source_w = source_img.shape[:2]
        target_img = np.zeros_like(source_img)

        for channel in range(CHANNELS):
            for i in range(source_h):
                for j in range(source_w):
                    target_img[i][j][channel] = model[channel][source_img[i][j][channel]]

        (Image.fromarray(target_img, mode=mode)).save('{corrected}')
