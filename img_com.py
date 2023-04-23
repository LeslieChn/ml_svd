#!/usr/bin/env python3

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from project import SVD


def main(img_path):
    with Image.open(img_path) as img:
        imggray = img.convert('LA')
        img_matrix = np.array(list(imggray.getdata(band=0)), dtype=np.uint8)
        img_matrix.shape = (imggray.size[1], imggray.size[0])
        img_matrix = np.matrix(img_matrix)
        U, sigma, V = np.linalg.svd(img_matrix)
        trace = np.sum(sigma)
        ff = 0
        rank = {}

        # print("size", np.size(sigma))
        # print("sum", np.sum(sigma))
        for i in range(np.size(sigma)):
            ff += sigma[i]
            percentage = int((ff / trace)*100)
            # if (percentage in rank):
            #     continue
            if (i == 0 or i % 10 == 9):
                rank[i] = percentage
        # print(rank)
        for k, v in rank.items():
            print(f"rank: {k+1}, percentage: {v}")
            reconstimg = (np.matrix(U[:, :k+1]) *
                          np.diag(sigma[:k+1]) * np.matrix(V[:k+1, :])).astype(np.uint8)
            Image.fromarray(reconstimg).save(
                f"img/rank{k+1}_{v}%.tiff", 'tiff')


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compress an image and save outputs in img directory, using SVD")
    parser.add_argument("image_path",
                        type=str,
                        metavar='<image path>', help='Image path')
    args = parser.parse_args()

main(args.image_path)
