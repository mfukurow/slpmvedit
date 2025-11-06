from scipy.io import loadmat
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


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
    rlt_t = t_pulse[indice]
    return rlt, rlt_t


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
    idx_frame = tpulse2idx(t_min, t_max, path_trgmat)[0]

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
                xx = int(csvdata.loc[frame_current, b + ".x"])
                yy = int(csvdata.loc[frame_current, b + ".y"])
                color = tuple(int(c) for c in colors[ii])
                cv2.circle(frame, (xx, yy), 3, color, -1)

            out.write(frame)

        frame_current += 1

    cap.release()
    out.release()
    print("done!")


def mklabelmovie_v1(
    t_min: float,
    t_max: float,
    path_trgmat: str,
    path_csv: str,
    path_mv: str,
    path_outmv: str,
    tpts: np.ndarray,
    x: np.ndarray,
    x_name: str,
    win_disp: float,
) -> None:
    """
    generate a SLEAP-labeled video with a time series data

    Args:
        t_min (float): minimum time
        t_max (float): maximum time
        path_trgmat (str): filepath for trigger mat
        path_csv (str): filepath for sleap csv
        path_mv (str): filepath for input movie
        path_outmv (str): filepath for output movie
        tpts (np.ndarray): time points for 'x'
        x (np.ndarray): time series data for display
        x_name (str): y label for variable 'x'
        win_disp (float): time window for display (sec)
    """

    # get indice information
    idxtinfo = tpulse2idx(t_min, t_max, path_trgmat)
    idx_frame = idxtinfo[0]
    t_frame = idxtinfo[1]

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

    # size for video
    h_out = int(h * 4 / 3)
    h_plot = h_out - h

    # set output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path_outmv, fourcc, fps, (w, h_out))

    # make a canvas
    fig, axes = plt.subplots(1, 1, figsize=(w / 100, h_plot / 100), dpi=100)
    canvas = FigureCanvas(fig)
    (line_x,) = axes.plot([], [])
    axes.set_xlabel("Time (s)")
    axes.set_ylabel(x_name)
    (line_v,) = axes.plot([t_frame[0], t_frame[0]], [-1, 1], "k", linewidth=2)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)

    # run
    frame_current = 0
    ii = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_current in idx_frame:
            # subset data
            twin_min = t_frame[ii] - win_disp / 2
            twin_max = t_frame[ii] + win_disp / 2
            idx_sub = np.where((tpts >= twin_min) & (tpts <= twin_max))
            tpts_sub = tpts[idx_sub]
            x_sub = x[idx_sub]
            xsub_amp = x_sub.max() - x_sub.min()
            ymin = x_sub.min() - xsub_amp * 0.1
            ymax = x_sub.max() + xsub_amp * 0.1

            # update plot
            line_x.set_data(tpts_sub, x_sub)
            axes.set_xlim(twin_min, twin_max)
            axes.set_ylim(ymin, ymax)
            line_v.set_data([t_frame[ii], t_frame[ii]], [ymin, ymax])
            canvas.draw()

            # transform plot into numpy array
            plot_img = np.array(canvas.renderer.buffer_rgba())
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
            plot_img = cv2.resize(plot_img, (w, h_plot))

            # update video picture
            for jj, b in enumerate(bodyparts):
                xx = int(csvdata.loc[frame_current, b + ".x"])
                yy = int(csvdata.loc[frame_current, b + ".y"])
                color = tuple(int(c) for c in colors[jj])
                cv2.circle(frame, (xx, yy), 3, color, -1)

            # combine
            combined_img = np.vstack((frame, plot_img))

            # output
            out.write(combined_img)

            ii += 1

        frame_current += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
