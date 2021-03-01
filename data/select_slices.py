import numpy as np
import os
import h5py
import cv2 as cv
import argparse


class SelectSlices:
    def __init__(self, volume_dir, nb_slices, output_dir):
        self.volume_dir = volume_dir
        self.nb_slices = nb_slices
        assert self.nb_slices%2 != 0, "need an odd number of slices"
        self.output_dir = output_dir

    def read_file_names(self):
        """
        read automatically the the volume file_name
        """
        files_volume = os.listdir(self.volume_dir)
        return files_volume

    def slice_volume(self, volume):
        """
        Take the volume as a numpy array and select nb_slices around the center of the volume
        """
        thickness = volume.shape[0]
        center = int(round(thickness/2))
        radius = int((self.nb_slices-1)/2)
        selected_slices = volume[(center-radius):(center+radius+1), :, :]
        return selected_slices

    def check_output_directory(self):
        """
        Check if the output directory already exist, otherwie create it
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def save_2d_slices(self):
        # check the existence of the output directory
        self.check_output_directory()
        # get the list of the files volume
        files_volume = self.read_file_names()

        # iterate over the files
        for file in files_volume:
            path = os.path.join(self.volume_dir, file)
            data = h5py.File(path, 'r')
            volume = np.array(data.get('reconstruction_rss'))
            slices = self.slice_volume(volume)
            for i, slice in enumerate(slices):
                slice = 255*((slice - np.amin(slice)) / (np.amax(slice) - np.amin(slice)))
                cv.imwrite(f'{self.output_dir}/{file[:-3]}_{i}.png', slice)


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='generate 2d slices')
    parser.add_argument('-vd', '--volumedir', default="singlecoil_train", type=str,
                        help='directory of the volume files')
    parser.add_argument('-od', '--outputdir', default="singlecoil_train_5_2d", type=str,
                        help='output directory in which slices are saved')
    parser.add_argument('-ns', '--nbslices', default=5, type=int,
                        help='number of slices to extract from the volume')
    args = parser.parse_args()

    generate_slices = SelectSlices(args.volumedir, args.nbslices, args.outputdir)
    generate_slices.save_2d_slices()
