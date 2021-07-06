'''
Source from: https://blog.csdn.net/QiaXi/article/details/52146526
Modified by bliu3650
'''

import cv2
import numpy as np
import os
import struct
import sys
import zipfile


def gnt2png(src_file, image_size, tgt_folder):
    if zipfile.is_zipfile(src_file): 
        zip_file = zipfile.ZipFile(src_file, 'r')
        file_list = zip_file.namelist()     
        for file_name in file_list:
            print('processing {} ...'.format(file_name))
            data_file = zip_file.open(file_name)
            total_bytes = zip_file.getinfo(file_name).file_size
            readFromGnt(
                tgt_folder,
                data_file,
                file_name,
                image_size,
                total_bytes
            )
    else:
        sys.stderr.write('Source file should be a ziped file containing '
                         'the gnt files. Plese check your input again.')
        return None


def readFromGnt(tgt_folder, data_file, file_name, image_size, total_bytes):
    decoded_bytes = 0
    while decoded_bytes != total_bytes:
        data_length, = struct.unpack('<I', data_file.read(4))
        tag_code, = struct.unpack('>H', data_file.read(2))
        image_width, = struct.unpack('<H', data_file.read(2))
        image_height, = struct.unpack('<H', data_file.read(2))
        arc_length = image_width
        if image_width < image_height:
            arc_length = image_height
        
        temp_image = 255 * np.ones((arc_length, arc_length ,1), np.uint8)
        row_begin = (arc_length - image_height) // 2
        col_begin = (arc_length - image_width) // 2
        for row in range(row_begin, image_height + row_begin):
            for col in range(col_begin, image_width + col_begin):
                temp_image[row, col], = struct.unpack('B', data_file.read(1))
        
        decoded_bytes += data_length
        # partial of Gnt1.2Test.zip/767-f.gnt had issues to be extracted!
        # manually zip left gnts (768-800) and rerun this script
        result_image = cv2.resize(temp_image, (image_size, image_size))
        result_image_path = os.path.join(tgt_folder,
            str(file_name) + '_' + str(tag_code) + '.png')
        
        cv2.imwrite(result_image_path, result_image)

    return None


if __name__=='__main__':

    if len(sys.argv) < 4:
        sys.stderr.write('Please specify source file, '
            'image size and target folder. \n')
        sys.exit()

    src_file = sys.argv[1]
    image_size = int(sys.argv[2])
    tgt_folder = sys.argv[3]

    if not ((os.path.isfile(src_file)) and (os.path.isdir(tgt_folder))):
        raise FileNotFoundError(
            "Expected source file and target folder."
        )

    gnt2png(src_file, image_size, tgt_folder)
