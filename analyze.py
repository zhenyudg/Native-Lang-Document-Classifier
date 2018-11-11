import sys
import csv
import glob

from sklearn.linear_model import Perceptron
from sklearn.feature_extraction.text import CountVectorizer


class Response:
    filename = None #string
    tokenized_text = None #string
    prompt_number = None #int
    native_lang = None #string
    predicted_native_lang = None #string


def unpack_data_set(path_to_set):
    """
    :param path_to_set: Path (string) to the dataset
    :return: A list of tuples
    """
    data = list()

    possible_index_files = glob.glob(path_to_set + "/data/text/index-*.csv")
    if len(possible_index_files) == 0:
        print("No index-*.csv found in " + path_to_set + "/data/text/")
        return
    path_index_csv = possible_index_files[0]

    path_tokenized_responses = path_to_set + "/data/text/responses/tokenized/"
    with open(path_index_csv, newline="") as index_csvfile:
        csvreader = csv.reader(index_csvfile)
        for row in csvreader:
            assert len(row) == 4
            #When adding feature fields to Responses, don't forget to fill in the fields here!
            response = Response()
            response.filename = row[0]
            with open(path_tokenized_responses + response.filename) as tokenized_response_file:
                response.tokenized_text = tokenized_response_file.read()
            response.prompt_number = int(row[1][-1])
            response.native_lang = row[2]
            data.append(response)

    return data


def write_output_csv(test_set, output_file_name):
    """
    :param test_set: A list of Response objects that are part of the test set
    :param predictions: A list of predicted native languages (in string format)
    :param output_file_name: Path to the output csv file to be created
    :return:
    """
    assert len(test_set) == len(predictions)
    with open(output_file_name, mode="w") as outfile:
        csvwriter = csv.writer(outfile)
        for r in test_set:
            csvwriter.writerow([r.filename, r.predicted_native_lang])



if len(sys.argv) != 5:
    print("Usage: python3 analyze.py PATH_TO_TRAIN_SET PATH_TO_DEV_SET PATH_TO_TEST_SET OUTPUT_CSV_FILE")

path_train_set = sys.argv[1]
path_dev_set = sys.argv[2]
path_test_set = sys.argv[3]
output_csv_file = sys.argv[4]

print("Loading training & test sets")
train_set = unpack_data_set(path_train_set)
test_set = unpack_data_set(path_test_set)


print("Extracting features from training set")
vectorizer = CountVectorizer()

train_tokenized_text = [r.tokenized_text for r in train_set]
train_X = vectorizer.fit_transform(train_tokenized_text)

train_y = [r.native_lang for r in train_set]

print("Training classifier")
classifier = Perceptron()
classifier.fit(train_X, train_y)

print("Extracting features from test set")
test_tokenized_text = [r.tokenized_text for r in test_set]
test_X = vectorizer.transform(test_tokenized_text)

print("Making predictions")
predictions = classifier.predict(test_X)

assert len(predictions) == len(test_set)
for i in range(len(test_set)):
    test_set[i].predicted_native_lang = predictions[i]

write_output_csv(test_set, output_csv_file)