import argparse
import os
import shutil
import random


class SplitValTest:
    def __init__(self, volume_dir,valdir, testdir, number_val, number_test,seed):
        self.volume_dir = volume_dir
        self.val_dir = valdir
        self.test_dir = testdir
        self.number_val = number_val
        self.number_test = number_test
        assert (self.number_val + self.number_test) == 199
        self.seed = seed

    def read_file_names(self):
        """
        read automatically the the volume file_name
        """
        files_volume = os.listdir(self.volume_dir)
        return files_volume

    def check_output_directory(self):
        """
        Check if the output directory already exist, otherwise create it
        """

        if not os.path.exists(self.val_dir):
            os.makedirs(self.val_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

    def split_val(self):
        # check the existence of the output directory
        self.check_output_directory()
        # get the list of the files volume
        files_volume = self.read_file_names()
        random.seed(self.seed)
        random.shuffle(files_volume)

        for i, file in enumerate(files_volume):
            source = os.path.join(self.volume_dir, file)
            if i < self.number_val:
                destination = os.path.join(self.val_dir, file)
                shutil.copy(source, destination)
            else:
                destination = os.path.join(self.test_dir, file)
                shutil.copy(source, destination)


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='generate validation and test set from the original validation')
    parser.add_argument('-vd', '--volumedir', default="singlecoil_val", type=str,
                        help='directory of the volume files')
    parser.add_argument('-odv', '--outputdirval', default="singlecoil_validation", type=str,
                        help='output directory in which slices are saved')
    parser.add_argument('-odt', '--outputdirtest', default="singlecoil_test", type=str,
                        help='output directory in which slices are saved')
    parser.add_argument('-nv', '--numberval', default=100, type=int,
                        help='number of files inlcuded in the validation set')
    parser.add_argument('-nt', '--numbertest', default=99, type=int,
                        help='number of files included in the test set')
    parser.add_argument('-s', '--seed', default=42, type=int,
                        help='seed for shuffling the files')
    args = parser.parse_args()

    generate_sets = SplitValTest(args.volumedir, args.outputdirval, args.outputdirtest, args.numberval, args.numbertest,
                                 args.seed)
    generate_sets.split_val()
