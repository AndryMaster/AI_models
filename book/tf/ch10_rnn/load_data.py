import matplotlib.pyplot as plt
import numpy as np
import csv


def load_series(filename, series_idx=1):
    try:
        with open(filename) as csv_file:
            reader = csv.reader(csv_file)
            data = [float(row[series_idx]) for row in reader if len(row) > 0]
            normalized_data = (data - np.mean(data)) / np.std(data)
            return normalized_data
    except IOError:
        return None


def split_data(data, percent_train=0.8):
    num_rows = round(len(data) * percent_train)
    return data[:num_rows], data[num_rows:]


if __name__ == '__main__':
    time_series = load_series('international-airline-passengers.csv')
    print(np.shape(time_series), time_series)

    # train, test = split_data(time_series)

    plt.figure()
    plt.plot(time_series)
    plt.show()
