import numpy as np

def main():
    rep = np.load("checkpoints_byol/representations/projections_best.npy")
    print(rep.shape)

if __name__ == "__main__":
    main()