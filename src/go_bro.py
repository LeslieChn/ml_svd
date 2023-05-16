#!/usr/bin/env python3

from argparse import ArgumentParser
import numpy as np
import time


def cosine_similarity(vector1, vector2):
    dot = vector1*np.transpose(vector2)
    cosine = dot[0]/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
    return cosine


def knn(train_data, test_usr, k, distance_fn):
    neighbor_distances_and_indices = []
    for index, training_usr in enumerate(train_data):
        distance = distance_fn(training_usr, test_usr)

        neighbor_distances_and_indices.append((distance, index))

    sorted_neighbor_distances_and_indices = sorted(
        neighbor_distances_and_indices, reverse=True)

    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]

    return [kn[1] for kn in k_nearest_distances_and_indices]


def main(data_set: str, k_value: int, c_value: int):
    start_time = time.time()
    data = np.loadtxt(open(data_set, "rb"),
                      delimiter=",")
    # split dataset into trainning and test
    row_num = (data.shape[0])
    training_row = int(3*row_num/4)
    trainning_set = data[0:training_row]
    test_set = data[training_row:]

    u, s, v = np.linalg.svd(trainning_set)
    rank_v = np.linalg.matrix_rank(v)
    rank_u = np.linalg.matrix_rank(u)
    # print("Rank:", min(rank_u, rank_v))
    v_tran = np.transpose(v)
    total_error = 0
    # transform the test_user to truncated format
    for i in range(test_set.shape[0]):
        test_user_orig = np.matrix(test_set[[i], :])
        # print(test_user_orig)
        test_user_new = np.matrix(test_set[[i], :]) * \
            np.matrix(v_tran[:, :c_value]) * \
            np.linalg.inv(np.diag(s[:c_value]))

        res = knn(np.matrix(u[:, :c_value]), test_user_new,
                  k_value, cosine_similarity)
        reconstruct = (np.matrix(u[res, :c_value]) *
                       np.diag(s[:c_value]) * np.matrix(v[:c_value, :]))
        # print("reconstruct", reconstruct)
        prediction = reconstruct.mean(0)
        # print("prediction", prediction)
        error = 0
        review_count = 0
        for j in range(test_user_orig.shape[1]):
            if (test_user_orig[0, j] != 0):
                error += abs(test_user_orig[0, j] - prediction[0, j])
                review_count += 1
        # print("error/review_count", error/review_count)
        total_error += error/review_count
    print(f"The execution time is: {time.time() - start_time} seconds")
    print("Error", total_error/test_set.shape[0])


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Go brrr with the seating charts")
    parser.add_argument("data_set",
                        type=str,
                        metavar='<data set>', help='Dataset path')
    parser.add_argument("k_value",
                        type=int,
                        metavar='<k_value>', help='K value')
    parser.add_argument("c_value",
                        type=int,
                        metavar='<truncate value>', help='Truncate percentage 0-100')
    args = parser.parse_args()

    main(args.data_set, args.k_value, args.c_value)
