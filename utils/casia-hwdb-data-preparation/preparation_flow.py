'''
The reference flow for casia-hwdb2x data preprocessing for model training
Created by bliu3650
'''

import sys
import os
import codecs

def map_code_to_char(code: str):
    if not (len(code) == 4):
        raise AssertionError(code)
    last_two = code[2] + code[3]
    if last_two == '00':
        char = codecs.decode(code, 'hex_codec').decode('utf-16')
    else:
        char = codecs.decode(code, 'hex_codec').decode('gbk')

    return char


def map_codes_to_chars(codes_list: list):
    chars_list = []
    for code in codes_list:
        char = map_code_to_char(code)
        chars_list.append(char)

    return chars_list


def generate_char_img_gt(img_path:str, out_file_path: str):
    '''
    parse the lable code from hwdb1x image filename,
    construct the image and groundtruth file,
    and return the list of classes of codes.
    '''
    hwdb1x_codes_list = []
    image_gt_codes_file = open(out_file_path, 'w')
    for image in os.listdir(img_path):
        print(image)
        code_str_dec = image.split('_')[-1].split('.')[0]
        code_str_hex = hex(int(code_str_dec)).upper().split('X')[-1]
        image_gt_codes_file.write(
            os.path.join(img_path, image) + ',' + code_str_hex + '\n'
        )
        if code_str_hex not in hwdb1x_codes_list:
            hwdb1x_codes_list.append(code_str_hex)
    image_gt_codes_file.close()

    return hwdb1x_codes_list


def generate_text_img_gt(data_path: str, out_file_path: str):
    '''
    map codes of lables to chars,
    construct the image and groundtruth file,
    and return the list of classes of codes.
    '''
    codes_list = []
    image_gt_chars_file = open(out_file_path, 'w')
    for label_code_f in os.listdir(data_path):
        if not label_code_f.endswith('.txt'):
            continue
        print(label_code_f)
        img_filename = label_code_f.replace('txt', 'png')
        if not os.path.isfile(os.path.join(data_path, img_filename)):
            continue
        image_gt_chars_file.write(img_filename + ',')

        with open(os.path.join(data_path, label_code_f), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                if line == 'FFFF':
                    continue
                # 'gbk' codec can't decode byte 0xfd in position 0
                # wrong lable in icdar2013 competition set
                if line == 'FDA3':
                    line = 'A3FD'
                char = map_code_to_char(line)
                image_gt_chars_file.write(char)
                if line not in codes_list:
                    codes_list.append(line)

        image_gt_chars_file.write('\n')
    image_gt_chars_file.close()

    return codes_list


def generate_codes_list(hwdb1x_codes_list: list,
                        hwdb2x_train_codes_list: list,
                        hwdb2x_test_codes_list: list):
    '''
    combine lists into one
    '''
    codes_list = hwdb1x_codes_list
    for code in hwdb2x_train_codes_list:
        if code not in codes_list:
            codes_list.append(code)

    for code in hwdb2x_test_codes_list:
        if code not in codes_list:
            codes_list.append(code)

    return codes_list


def save_list_to_file(input_list: list, out_file_path: str):
    '''
    save input list into txt file with one element per line
    '''
    out_file = open(out_file_path, 'w')
    for item in input_list:
        out_file.write(item + '\n')
    out_file.close()

    return None


def select_alpha_symbol_codes(codes_list):
    '''
    alphanumeric and symbols (utf-8: xx00, gbk: A1xx-A9xx)
    '''
    alpha_symbol_codes_list = []
    codes_range = ['A1', 'A2', 'A3',
                   'A4', 'A5', 'A6',
                   'A7', 'A8', 'A9'
    ]
    for code in codes_list:
        first_two = code[0] + code[1]
        if first_two in codes_range:
            alpha_symbol_codes_list.append(code)
            continue
        last_two = code[2] + code[3]
        if last_two == '00':
            alpha_symbol_codes_list.append(code)
        
    return alpha_symbol_codes_list


def preparation_flow():
    # generate hwdb1x images-lables(codes/chars) file
    # and return the codes list
    hwdb1x_codes_list = generate_char_img_gt(
        './extracted_hwdb1x_data',
        'hwdb1x_img_gt_codes.txt'
    )

    # generate hwdb2x-train/test images-lables(chars) file
    # and return the codes list
    hwdb2x_train_codes_list = generate_text_img_gt(
        './extracted_hwdb2x_train_data',
        'hwdb2x_train_img_gt.txt'
    )

    hwdb2x_test_codes_list = generate_text_img_gt(
        './extracted_hwdb2x_test_data',
        'hwdb2x_test_img_gt.txt'
    )

    # generate icdar2013 competition set images-lables(chars) file
    icdar2013_codes_list = generate_text_img_gt(
        './extracted_icdar2013_comp_data',
        'icdar2013_comp_img_gt.txt'
    )

    # combine hwdb1x and hwdb2x codes list
    # and map to chars list for model training
    hwdb_codes_list = generate_codes_list(
        hwdb1x_codes_list,
        hwdb2x_train_codes_list,
        hwdb2x_test_codes_list
    )
    print('### len of hwdb codes list: {}'.format(len(hwdb_codes_list)))
    # 7373

    hwdb_chars_list = map_codes_to_chars(hwdb_codes_list)
    save_list_to_file(hwdb_chars_list,
        'hwdb_chars_list.txt'
    )

    # select alphanumeric and symbol codes from hwdb2x_train_codes_list
    # these codes (small char-image) will be reserved during the synthesis
    alpha_symbol_codes_list = select_alpha_symbol_codes(hwdb2x_train_codes_list)
    save_list_to_file(alpha_symbol_codes_list,
        'selected_alpha_symbol_codes.txt'
    )


if __name__=='__main__':
    if len(sys.argv) < 2:
        # NOTE: Extract images/lables from CASIA-HWDB1x/2x databases
        # gnt2png.py:
        #   hwdb1x images
        # dgr2png (bin):
        #   hwdb2x-train/test images/lables
        #   ICDAR2013 competition set images/lables
        
        # check data folder
        if not (os.path.isdir('./extracted_hwdb1x_data') and 
                os.path.isdir('./extracted_hwdb2x_train_data') and
                os.path.isdir('./extracted_hwdb2x_test_data') and
                os.path.isdir('./extracted_icdar2013_comp_data')):
            raise FileNotFoundError('CASIA-HWDB Data is required.')

        preparation_flow()
    elif (len(sys.argv) == 2) and (sys.argv[1] == 'synthesize'):
        # NOTE: Call dgr2png (bin) with synthesis mode
        # input:
        #   hwdb2x_train_dgrs.txt
        #   hwdb1x_img_gt_codes.txt
        #   selected_alpha_symbol_codes.txt
        # output:
        #   synthesized_data

        # check data folder
        if not (os.path.isdir('./synthesized_data')):
            raise FileNotFoundError('Synthesized data is required.')

        # generate synthesized data images-lables(chars) file:
        generate_text_img_gt('./synthesized_data', 'synthesized_img_gt.txt')
    else:
        sys.stderr.write('With or without parameter: synthesize is expected. \n')
        sys.exit()


# data strategy for training/validation/test
# chars_list.txt:
#   hwdb_chars_list.txt
# train_img_id_gt.txt:
#   hwdb2x_train_img_gt.txt
#   synthesized_img_gt.txt
# val_img_id_gt.txt:
#   hwdb2x_test_img_gt.txt
# test_img_id_gt.txt:
#   icdar2013_comp_img_gt.txt
#   NOTE: replacement alphanumeric and symbol
#   chars from chinese-input to english-input