import csv
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from os import listdir
import sklearn
from scipy.spatial import distance
import matplotlib.pyplot as plt
import shutil
from PIL import Image
import PIL

def load_files(files):
    response_loaded_files = []
    for file in files:
        file_loaded = scipy.io.loadmat(file)
        response_loaded_files.append(file_loaded)
    return response_loaded_files


def load_file(name):
    file_loaded = scipy.io.loadmat(name)
    return file_loaded


def split(file_array, row_limit=1024):
    result_list = []
    for mat_file in file_array:
        column_dataset = mat_file["X097_DE_time"]
        result = np.array_split(column_dataset, math.floor(column_dataset.shape[0] / row_limit))
        result_list.append(result)
    return result_list


def split_individual_file(file_array, row_limit=1024, column_name="DE"):
    for elem in file_array.keys():
        if elem.__contains__(column_name):
            column = elem
    column_dataset = file_array[column]
    result = np.array_split(column_dataset, math.floor(column_dataset.shape[0] / row_limit))
    return result


def create_matrix_distance(step, base_list):
    response_arrays = []
    for list in base_list:
        for array_based in list:
            x_value = array_based[step:].copy()
            y_value = array_based[:-step]
            new_matrix = np.column_stack((x_value, y_value))
            matrix = scipy.spatial.distance.cdist(new_matrix, new_matrix)
            response_arrays.append(matrix)
    return response_arrays


def create_matrix_distance_for_file(step, array_list):
    response_arrays = []
    for array_splitted in array_list:
        x_value = array_splitted[step:].copy()
        y_value = array_splitted[:-step]
        new_matrix = np.column_stack((x_value, y_value))
        matrix_distance = scipy.spatial.distance.cdist(new_matrix, new_matrix, metric='euclidean')
        response_arrays.append(matrix_distance)
    return response_arrays


def create_binary_matrix(distance_matrix, epsylon):
    binary_matrix_list = []
    for matr in distance_matrix:
        binary_matrix = (np.where(matr > epsylon, 1, 0))
        binary_matrix_list.append(binary_matrix)
    return binary_matrix_list


def create_binary_matrix_for_file(distance_matrix, epsylon):
    binary_matrix_list = []
    for matr in distance_matrix:
        binary_matrix = (np.where(epsylon> matr, 1, 0))
        binary_matrix_list.append(binary_matrix)
    return binary_matrix_list



def save_matrix_list_to_images(matrix_list, folder, prefix_name):
    index = 0
    for element in matrix_list:
        image = PIL.Image.fromarray(np.uint8(element*255))
        image = image.resize((256,256), Image.BICUBIC)
        image.save(folder + '/{0}_fig{1}.png'.format(prefix_name, str(index)))
        #fig = plt.figure(frameon=False)
        #fig.set_size_inches(w, h)
        #ax = plt.Axes(fig, [0., 0., 1., 1.])
        #ax.set_axis_off()
        #fig.add_axes(ax)
        #plt.imshow(element)
        #plt.savefig(folder + '/{0}_fig{1}.png'.format(prefix_name, str(index))
        # plt.savefig("./test/output_{0}.png".format(str(index)), bbox_inches='tight')
        index += 1


def binary_matrix_list_to_rgb(matrix_list, folder):
    index = 0
    for element in matrix_list:
        img = Image.new('1', element.shape)
        pixels = img.load()
        #plt.imshow(element)
        #plt.savefig(folder + '/fig{0}.png'.format(str(index)))
        # plt.savefig("./test/output_{0}.png".format(str(index)), bbox_inches='tight')
        index += 1

def process_bunch_files():
    '''

    :return:
    '''
    directory = "./test"
    if not os.path.exists(directory):
        os.makedirs(directory)
    newarray = ['/Users/lisandro/Downloads/97.mat']
    array_of_files = load_files(newarray)
    result = split(array_of_files, row_limit=128)
    full_matrix_distance = create_matrix_distance(2, result)
    binary_matrix_list = create_binary_matrix(full_matrix_distance, 0.05)
    save_matrix_list_to_images(binary_matrix_list, directory)

def process_individual_file(input_file, output_folder, column="DE", name_prexix= "", row_limit=128, epsylon=0.05, distance_matrix=2):
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    file_loaded = load_file(input_file)
    splitted_file = split_individual_file(file_loaded, row_limit=row_limit, column_name=column) #need to send column, probably will have to get it from filename
    full_matrix_distance = create_matrix_distance_for_file(distance_matrix, splitted_file)
    binary_matrix_list = create_binary_matrix_for_file(full_matrix_distance, epsylon)
    save_matrix_list_to_images(binary_matrix_list, output_folder, name_prexix)




def process_all(main_folder="/Users/lisandro/Downloads/engine_data", data_type="DE", points_per_file=1024, epsylon=0.5, distance_point=2, points_per_distance=2):
    for root, dirs, files in os.walk(main_folder, topdown=False):
        for file_name in files:
            if file_name.endswith('.mat'):
                folder = os.path.join(root, os.path.splitext(file_name)[0])
                if os.path.exists(folder):
                    shutil.rmtree(folder)
                os.makedirs(folder)
                if folder.__contains__("inner_race"):
                    prefix= 'innerrace'
                else:
                    if folder.__contains__("baseline"):
                        prefix = 'baseline'
                    else:
                        prefix='ball'
                if folder.__contains__("DE"):
                    data_type="DE"
                else:
                    if folder.__contains__("FE"):
                        data_type="FE"
                    else:
                        os.makedirs(folder+ "FE")
                        process_individual_file(root + file_name, folder + "FE", data_type, prefix, points_per_file, epsylon,
                                                distance_point)
                        os.makedirs(folder + "DE")
                        data_type = "DE"

                process_individual_file(root + "/" + file_name, folder, data_type, prefix, points_per_file, epsylon, distance_point)

process_all()
#process_individual_file("/Users/lisandro/Downloads/engine_data/baseline/baseline_hp_0_rpm_1797.mat", "/Users/lisandro/Downloads/engine_data/baseline/baseline_hp_0_rpm_1797", "X097_DE_time", "baseline")
#process_individual_file("/Users/lisandro/Downloads/engine_data/DE_fault/DE_fault_007_hp_0_rpm_1797_ball.mat", "/Users/lisandro/Downloads/engine_data/DE_fault/DE_fault_007_hp_0_rpm_1797_ball", "X118_DE_time", "ball")
#process_individual_file("/Users/lisandro/Downloads/engine_data/DE_fault/DE_fault_007_hp_0_rpm_1797_inner_race.mat", "/Users/lisandro/Downloads/engine_data/DE_fault/DE_fault_007_hp_0_rpm_1797_inner_race", "X105_DE_time", "innerrace",128)

