import os
from got10k.datasets import *
from got10k.experiments import *
from models.siamfc import TrackerSiamFC

if __name__ == '__main__':
    data_dir = os.path.expanduser('/Users/coder/project/UWOT/datasets/GOT10k')
    seqs = GOT10k(data_dir, subset='train', return_meta=True)
    tracker = TrackerSiamFC()
    tracker.train_over(seqs)
