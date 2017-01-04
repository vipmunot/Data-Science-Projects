
# NOTE: The program is python 2.7 compatible. Kindly run on compatible python version to avoid incorrect results/errors.
#
# This is the Naive Bayes Code:
# Please run this file with the following command:
# python NB_AML2.py data_file, delimiter, header_present_flag, label_index, fraction

# Example:
# 	python NB_AML2.py "default of credit card clients_no_headers.csv" "," False 15 0.1

# The parameters for all the 3 datasets are given below

# It takes 5 parameters as follows:
# 1) data_file = The data file name (with the full path if the file is not in the same folder as the python file)
# 	eg: "default of credit card clients_no_headers.csv"
#
# 2) delimiter = The delimiter used to separate the columns in the data (Can be "," or blankspace(" ") or anything else
# as long as its consistent through out the file)
# 	eg: ","
#
# 3) header_present_flag = True/False.Used to notify whether the data file contains the first row as column header or not
# 	eg: False
#
# 4) label_index = Need to give the index of the label column. (0,1,2....
#    NOTE: Index starts at 0 and NOT 1 !!!
# 		eg: 15
#
# 5) fraction: The fraction (should be always between 0 and 1) in which the data needs to be divided as training and test.
#    Eg: If 0.9 is given, then it randomly assigns 90% of the data as train and rest 10% as test.
# 		eg: 0.9

# Note: Please make sure you give data_file and header_present_flag in double quotes ("") and NOT single quotes ('')
#
# Parameters:
# 1) German Data Set (german.data)
# data_file = "german.data"
# delimiter = " " (Blank Space)
# header_present_flag = False
# Label Index = 20
# fraction = 0.9 (Can be changed to anything between 0 and 1)
#
# 2) CRX Data Set
# data_file = "crx.data"
# delimiter = "," (Comma)
# header_present_flag = False
# Label Index = 15
# fraction = 0.9 (Can be changed to anything between 0 and 1)
#
# 3) Default Data Set
# data_file = "default of credit card clients.csv"
# delimiter = "," (Comma)
# header_present_flag = True
# Label Index = 23
# fraction = 0.9 (Can be changed to anything between 0 and 1)


import os
import sys
import collections
from collections import defaultdict, Counter
import math as m
import time
import random
import pandas as pd

# Global Variables
distinct_label_values = set()
label_priors = Counter()
label_features_likelihood = defaultdict(Counter)


# Returns a coin flip probability based on the given fraction
def coin_flip(fraction):
    return random.random() >= 1 - fraction


def split_train_test_data(full_data, label_index, fraction):
    global distinct_label_values
    train_data = []
    test_data = []
    for row in full_data:
        distinct_label_values.add(row[label_index])
        train_data.append(row) if coin_flip(fraction) else test_data.append(row)
    return train_data, test_data


# Returns a list of words in a given document file
def get_all_data_list(file_path, delimiter, header_present, label_index):
    try:
        with open(file_path) as f:
            if 'german' in file_path:
                label_index = 20
            elif 'crx' in file_path:
                label_index = 15
            elif 'default' in file_path:
                label_index = 23
            full_data = [line.rstrip('\n').split(delimiter) for line in f]
            all_columns = []
            if header_present.lower() == "true":
                all_columns = full_data.pop(0)
                for col in range(len(all_columns)):
                    all_columns[col] = all_columns[col].strip()
            return all_columns, full_data, label_index
    except IOError:
        print("Cannot read from file:", file_path)
        return 0


# Reads the training data from a given file and creates a vocabulary, priors and likelihoods
def read_training_data(full_train_data, label_column_index):
    print("Reading the training data and creating a vocabulary, priors and likelihoods...")
    global label_priors, label_features_likelihood
    # label = all_columns[label_column_index]
    row_count = 0
    for distinct_label in distinct_label_values:
        label_priors[distinct_label] = 0
        for c in range(len(full_train_data[0])):
            if c != label_column_index:
                label_features_likelihood[distinct_label][c] = Counter()
    for row in full_train_data:
        row_count += 1
        row_label = row[label_column_index]
        label_priors[row_label] += 1
        for x in range(len(row)):
            if x != label_column_index:
                # if label_features_likelihood[row_label][all_columns[x]] == 0:
                #     label_features_likelihood[row_label][all_columns[x]] = Counter()
                label_features_likelihood[row_label][x][row[x]] += 1

# Trains the NB classifier
def nb_train_data(data, label_column_index):
    print("Training Naive Bayes Classifier...")
    print("Thank you for waiting...\nWe appreciate your patience...")
    # Reads the training data from a given file and creates a vocabulary, priors and likelihoods
    read_training_data(data, label_column_index)
    # Normalizing the likelihood counts ie. converting them to probabilities
    normalize_likelihood()

# Normalizing the likelihood ie. converting them into probabilities
def normalize_likelihood():
    global label_features_likelihood, label_priors
    for label in label_priors.keys():
        # total_words = float(sum(topic_word_likelihood[topic].values()))
        label_priors[label] /= 1.0 * sum(label_priors.values())
        for feature in label_features_likelihood[label].keys():
            total_features = sum(label_features_likelihood[label][feature].values())
            for value in label_features_likelihood[label][feature]:
                label_features_likelihood[label][feature][value] /= 1.0 * total_features


# Reads the test data from the given directory
def nb_test_data(full_test_data, label_column_index):
    print ("Testing the test data set...")
    test_file_count = 0
    test_correct_count = 0
    nb_confusion_matrix, all_label_list = create_confusion_matrix()
    # label = all_columns[label_column_index]
    for row in full_test_data:
        test_file_count += 1
        actual_row_label = row[label_column_index]
        predicted_row_label = naive_bayes(row, label_priors, label_features_likelihood, label_column_index)  # , nb_confusion_matrix)
        if actual_row_label == predicted_row_label:
            test_correct_count += 1
        nb_confusion_matrix[actual_row_label][predicted_row_label] += 1
        nb_confusion_matrix[actual_row_label]["Actual Count"] += 1

    print_confusion_matrix(nb_confusion_matrix, all_label_list)
    return calc_accuracy("nb_confusion_matrix", test_correct_count, test_file_count)
    # return (100.0 * test_correct_count)/test_file_count


# Performs Naive Bayes on a given doc_string
def naive_bayes(row, l_priors, l_f_likelihood, label_column_index):
    max_class = (-1E6, '')
    for label in l_priors:
        p_topic = m.log(l_priors[label])
        # p_topic = m.log(l_priors[label]/sum(l_priors[label].values()))
        # topic_total_words = float(sum(t_w_likelihood[topic].values()))
        for x in range(len(row)):
            if x != label_column_index:
                # print l_f_likelihood[label][all_columns[x]]
                p_topic += m.log(max(1E-6, l_f_likelihood[label][x][row[x]]))  # - m.log(topic_total_words)
        if p_topic > max_class[0]:
            max_class = (p_topic, label)

    return max_class[1]
    # return nb_confusion_matrix


# Calculates accuracy based on the given confusion matrix
def calc_accuracy(confusion_matrix, test_correct_count, test_file_count):
    return (100.0 * test_correct_count) / test_file_count


# Creates confusion matrix dictionary
def create_confusion_matrix():
    confusion_matrix = collections.OrderedDict()
    all_label_list = label_priors.keys()
    for topic in all_label_list:
        confusion_matrix[topic] = collections.OrderedDict()
        for all_topics in all_label_list:
            confusion_matrix[topic][all_topics] = 0
        confusion_matrix[topic]["Actual Count"] = 0
    return (confusion_matrix, all_label_list)


# Print confusion matrix
def print_confusion_matrix(confusion_matrix, all_label_list):
    print ("Confusion Matrix")
    print ("\t Model Results")
    all_label_list.append("Actual Count")
    confusion_list = [[all_label_list[y] for x in range(len(all_label_list) + 1)] for y in
                      range(len(all_label_list) - 1)]
    confusion_header = ["Actual\Model"] + all_label_list
    for i in range(len(confusion_header) - 2):
        for j in range(1, len(confusion_header)):
            confusion_list[i][j] = confusion_matrix[all_label_list[i]][confusion_header[j]]
    cm = pd.DataFrame(confusion_list, columns=confusion_header)
    print cm.to_string(index=False)
    print ("\n")


# Called to initialize the program
def main(data_file, delimiter, header_present_flag, label_index, fraction):
    all_columns, full_data, label_index = get_all_data_list(data_file, delimiter, header_present_flag, label_index)
    train_data, test_data = split_train_test_data(full_data, label_index, fraction)
    nb_train_data(train_data, label_index)
    accuracy = nb_test_data(test_data, label_index)
    print ("Naive Bayes Accuracy", float(accuracy))


if __name__ == '__main__':
    start_time = time.time()
    data_file = str(sys.argv[1])
    delimiter = sys.argv[2]
    header_present_flag = str(sys.argv[3])
    label_index = int(sys.argv[4])
    fraction = float(sys.argv[5])
    # main(data_file, delimiter, header_present_flag, label_index, fraction)
    main(data_file, delimiter, header_present_flag, label_index, fraction)

    end_time = time.time()
    print ("Total Time", end_time - start_time)
