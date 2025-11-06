from scipy.io import loadmat
import numpy as np
import pandas as pd
import cv2


def tpulse2idx(t_min: float, t_max: float, pathdata: str) -> int:
    """
    return video indice information based on time range (t_pulse)

    Args:
        t_min (float): minimum time
        t_max (float): maximum time
        pathdata (str): filepath to trig.mat

    Returns:
        int: indice information
    """
    dattrg = loadmat(pathdata)
    t_pulse = dattrg["t_pulse"].squeeze()
    is_trigframe = dattrg["is_trigframe"].squeeze().astype(bool)

    offset = np.where(is_trigframe)[0][0]
    indice = np.where((t_pulse >= t_min) & (t_pulse <= t_max))[0]

    rlt = indice + offset
    return rlt


def mklabelmovie(
    t_min: float,
    t_max: float,
    path_trgmat: str,
    path_csv: str,
    path_mv: str,
    path_outmv: str,
) -> None:
    """
    generate a SLEAP-labeled video trimmed to a user-defined time window

    Args:
        t_min (float): minimum time
        t_max (float): maximum time
        path_trgmat (str): filepath for trigger mat
        path_csv (str): filepath for sleap csv
        path_mv (str): filepath for input movie
        path_outmv (str): filepath for output movie
    """

    # get indice information
    idx_frame = tpulse2idx(t_min, t_max, path_trgmat)

    # load sleap csv data
    csvdata = pd.read_csv(path_csv)

    # body parts and colors
    bodyparts = [c[:-2] for c in csvdata.columns if c.endswith(".x")]
    N = len(bodyparts)
    colors = np.linspace(0, 255, N).astype(np.uint8)
    colors = cv2.applyColorMap(colors.reshape(-1, 1), cv2.COLORMAP_HSV)
    colors = colors.reshape(N, 3)

    # load video
    cap = cv2.VideoCapture(path_mv)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # set output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path_outmv, fourcc, fps, (w, h))

    # run
    frame_current = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_current in idx_frame:
            for ii, b in enumerate(bodyparts):
                x = int(csvdata.loc[frame_current, b + ".x"])
                y = int(csvdata.loc[frame_current, b + ".y"])
                color = tuple(int(c) for c in colors[ii])
                cv2.circle(frame, (x, y), 3, color, -1)

            out.write(frame)

        frame_current += 1

    cap.release()
    out.release()
    print("done!")
