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


def read_slpcsv(path_csv: str, path_mv: str) -> dict:
    """
    Load a SLEAP-exported CSV and corresponding movie file, then generate
    per-individual tracking data aligned to the full set of video frame indices.

    This function reads a SLEAP CSV file that contains tracking results
    (frame_idx, track ID, keypoint coordinates, etc.), extracts all unique
    individuals (track IDs), and constructs a clean DataFrame for each
    individual. Missing frames (i.e., frames where SLEAP did not detect
    the individual) are added and filled with NaN by reindexing against
    the total number of frames in the movie file.

    If only a single individual is present, the result is returned as
    output["track_0"]. If multiple individuals exist, the dictionary keys
    correspond to the original track IDs in the CSV.

    Args:
        path_csv (str): filepath for sleap csv
        path_mv (str): filepath for input movie

    Returns:
        dict: A dictionary mapping each individual/track to a DataFrame
            whose index corresponds to all movie frames (0 to n_frame-1),
            with missing frames filled with NaN.
    """

    # load data
    csvdata = pd.read_csv(path_csv)
    cap = cv2.VideoCapture(path_mv)

    # number of frames
    n_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = range(0, n_frame)

    # processing for each individual
    ids = csvdata["track"].unique()

    output = {}

    if len(ids) < 2:
        csvdata = csvdata.set_index("frame_idx").reindex(frame_idx)
        output["track_0"] = csvdata
    else:
        for id in ids:
            csvdata_id = csvdata[csvdata["track"] == id].copy()
            csvdata_id = csvdata_id.set_index("frame_idx").reindex(frame_idx)

            output[id] = csvdata_id

    return output


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
    idx_frame = set(tpulse2idx(t_min, t_max, path_trgmat)[0])

    # load sleap csv data
    slpcsvdata = read_slpcsv(path_csv, path_mv)
    ids = list(slpcsvdata.keys())

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

    if len(ids) < 2:
        # body parts and colors
        csvdata = slpcsvdata[ids[0]]
        bodyparts = [c[:-2] for c in csvdata.columns if c.endswith(".x")]
        N = len(bodyparts)
        colors = np.linspace(0, 255, N).astype(np.uint8)
        colors = cv2.applyColorMap(colors.reshape(-1, 1), cv2.COLORMAP_HSV)
        colors = colors.reshape(N, 3)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_current in idx_frame:
                for ii, b in enumerate(bodyparts):
                    xv = csvdata.loc[frame_current, b + ".x"]
                    yv = csvdata.loc[frame_current, b + ".y"]
                    if not np.isnan(xv) and not np.isnan(yv):
                        color = tuple(int(c) for c in colors[ii])
                        cv2.circle(frame, (int(xv), int(yv)), 3, color, -1)

                out.write(frame)

            frame_current += 1
            if frame_current % 1000 == 0:
                print(frame_current)

    else:
        # color for each individual
        N = len(ids)
        colors = np.linspace(0, 255, N).astype(np.uint8)
        colors = cv2.applyColorMap(colors.reshape(-1, 1), cv2.COLORMAP_JET)
        colors = colors.reshape(N, 3)
        xname = [c for c in slpcsvdata[ids[0]].columns if c.endswith(".x")]
        yname = [c for c in slpcsvdata[ids[0]].columns if c.endswith(".y")]

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_current in idx_frame:
                for ii, id in enumerate(ids):
                    xx = slpcsvdata[id].loc[frame_current, xname].to_numpy()
                    yy = slpcsvdata[id].loc[frame_current, yname].to_numpy()
                    for xi, yi in zip(xx, yy):
                        if not np.isnan(xi) and not np.isnan(yi):
                            color = tuple(int(c) for c in colors[ii])
                            cv2.circle(frame, (int(xi), int(yi)), 3, color, -1)

                out.write(frame)

            frame_current += 1
            if frame_current % 1000 == 0:
                print(frame_current)

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
    idx_frame = set(idxtinfo[0])
    t_frame = idxtinfo[1]

    # load sleap csv data
    slpcsvdata = read_slpcsv(path_csv, path_mv)
    ids = list(slpcsvdata.keys())
    csvdata = slpcsvdata[ids[0]]

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
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_current in idx_frame:
            # subset data
            twin_min = t_frame[i] - win_disp / 2
            twin_max = t_frame[i] + win_disp / 2
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
            line_v.set_data([t_frame[i], t_frame[i]], [ymin, ymax])
            canvas.draw()

            # transform plot into numpy array
            plot_img = np.array(canvas.renderer.buffer_rgba())
            plot_img = cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR)
            plot_img = cv2.resize(plot_img, (w, h_plot))

            # update video picture
            for ii, b in enumerate(bodyparts):
                xv = csvdata.loc[frame_current, b + ".x"]
                yv = csvdata.loc[frame_current, b + ".y"]
                if not np.isnan(xv) and not np.isnan(yv):
                    color = tuple(int(c) for c in colors[ii])
                    cv2.circle(frame, (int(xv), int(yv)), 3, color, -1)

            # combine
            combined_img = np.vstack((frame, plot_img))

            # output
            out.write(combined_img)

            i += 1

        frame_current += 1
        if frame_current % 1000 == 0:
            print(frame_current)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
