#!/usr/bin/env python3

from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main(img_path):
    with Image.open(img_path) as img:
        imggray = img.convert('LA')
        img_matrix = np.array(list(imggray.getdata(band=0)), dtype=np.uint8)
        img_matrix.shape = (imggray.size[1], imggray.size[0])
        img_matrix = np.matrix(img_matrix)
        U, sigma, V = np.linalg.svd(img_matrix)
        ss = np.sum(sigma)
        ff = 0
        rank = {}
        print("size", np.size(sigma))
        print("sum", np.sum(sigma))
        for i in range(np.size(sigma)):
            ff += sigma[i]
            percentage = int((ff / ss)*100)
            if (percentage in rank):
                continue
            rank[percentage] = i
        print(rank)
        c = np.percentile(sigma, 1, method='linear')
        # print(c)
        # for i in rank:
        #     c = np.percentile(sigma, i)
        #     print(c)
        #     c = int(np.size(sigma)*i/100)
        #     reconstimg = (np.matrix(U[:, :c]) *
        #                   np.diag(sigma[:c]) * np.matrix(V[:c, :])).astype(np.uint8)
        #     Image.fromarray(reconstimg).save(f"img/{i}%.tiff", 'tiff')


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Compress an image and save outputs in img directory, using SVD")
    parser.add_argument("image_path",
                        type=str,
                        metavar='<image path>', help='Image path')
    args = parser.parse_args()

main(args.image_path)
