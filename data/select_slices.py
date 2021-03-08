import numpy as np
import os
import h5py
import cv2 as cv
import argparse
import torch


class SelectSlices:
    def __init__(self, volume_dir, nb_slices, noise, output_dir, sigma):
        self.volume_dir = volume_dir
        self.nb_slices = nb_slices
        assert self.nb_slices % 2 != 0, "need an odd number of slices"
        self.noise = noise
        self.output_dir = output_dir + "_" + str(nb_slices) + "_" + str(noise) + "_" + str(sigma)
        self.path_images = os.path.join(self.output_dir, "Target_images")
        self.path_tensors = os.path.join(self.output_dir, "Target_tensors")
        self.sigma = sigma

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
        Check if the output directory already exist, otherwise create it
        """

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.path_images):
            os.makedirs(self.path_images)
        if not os.path.exists(self.path_tensors):
            os.makedirs(self.path_tensors)

        paths_images = []
        paths_tensors = []
        for i in range(self.noise):
            path_images = os.path.join(self.output_dir, f"Noise_images_{i+1}")
            paths_images.append(path_images)
            path_tensors = os.path.join(self.output_dir, f"Noise_tensors_{i+1}")
            paths_tensors.append(path_tensors)
            if not os.path.exists(path_images):
                os.makedirs(path_images)
            if not os.path.exists(path_tensors):
                os.makedirs(path_tensors)
        return paths_images, paths_tensors

    def save_2d_slices(self):
        # check the existence of the output directory
        paths_images, paths_tensors = self.check_output_directory()
        # get the list of the files volume
        files_volume = self.read_file_names()
        SNR = []
        # iterate over the files
        for file in files_volume:
            path = os.path.join(self.volume_dir, file)
            data = h5py.File(path, 'r')
            volume = np.array(data.get('reconstruction_rss'))
            slices = self.slice_volume(volume)
            for i, slice in enumerate(slices):
                slice = ((slice - np.amin(slice)) / (np.amax(slice) - np.amin(slice)))
                cv.imwrite(f'{self.path_images}/{file[:-3]}_{i}.png', 255*slice)
                slice_tensor = torch.from_numpy(slice)
                torch.save(slice_tensor, f'{self.path_tensors}/{file[:-3]}_{i}.pt')
                for j in range(self.noise):
                    noise = torch.randn(slice_tensor.shape)
                    noisy_slice = slice_tensor + noise * self.sigma
                    path_images = paths_images[j]
                    cv.imwrite(f'{path_images}/{file[:-3]}_{i}.png', 255 * noisy_slice.numpy())
                    path_tensors = paths_tensors[j]
                    torch.save(noisy_slice, f'{path_tensors}/{file[:-3]}_{i}.pt')
                    snr = 20*np.log10(np.linalg.norm(slice_tensor.numpy().flatten('F'))/np.linalg.norm(noisy_slice.numpy().flatten('F')-slice_tensor.numpy().flatten('F')))
                    SNR.append(snr)
        SNR = np.array(SNR)
        print("The mean SNR is {}".format(SNR.mean()))
        print("The max SNR is {}".format(np.amax(SNR)))
        print("The min SNR is {}".format(np.amin(SNR)))


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='generate 2d slices')
    parser.add_argument('-vd', '--volumedir', default="singlecoil_train", type=str,
                        help='directory of the volume files')
    parser.add_argument('-od', '--outputdir', default="singlecoil_train", type=str,
                        help='output directory in which slices are saved')
    parser.add_argument('-ns', '--nbslices', default=5, type=int,
                        help='number of slices to extract from the volume')
    parser.add_argument('-n', '--noise', default=2, type=int,
                        help='number of noisy realisation')
    parser.add_argument('-s', '--sigma', default=0.02, type=float,
                        help='std of the gaussian noise')
    args = parser.parse_args()

    generate_slices = SelectSlices(args.volumedir, args.nbslices, args.noise, args.outputdir, args.sigma)
    generate_slices.save_2d_slices()
