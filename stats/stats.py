# plotting script
import json
import os
import matplotlib
from scipy.stats import t
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams.update({'font.size': 20})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import colorConverter as cc
import numpy as np
from math import sqrt
from mpl_toolkits.axes_grid1 import make_axes_locatable
SEED = 0
SIZE = 1
DIM = 2
ITERATIONS = 3
MATCH_PERC = 4
THREADS = 5

LIST_ALGO = "list"
LOCK_ALGO = "lock"

SPEED_UP_BORDERS = (1.25, 4)
VAR_BORDERS = (0, 0.3)

def heatmap(data, row_labels, col_labels, borders, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, vmin=borders[0], vmax=borders[1])

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, orientation="horizontal")
    cbar.ax.set_ylabel(cbarlabel, va="bottom", labelpad=30)

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


class LegendObject(object):
    def __init__(self, facecolor='red', edgecolor='white', dashed=False):
        self.facecolor = facecolor
        self.edgecolor = edgecolor
        self.dashed = dashed

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        patch = mpatches.Rectangle(
            # create a rectangle that is filled with color
            [x0, y0], width, height, facecolor=self.facecolor,
            # and whose edges are the faded color
            edgecolor=self.edgecolor, lw=3)
        handlebox.add_artist(patch)

        # if we're creating the legend for a dashed line,
        # manually add the dash in to our rectangle
        if self.dashed:
            patch1 = mpatches.Rectangle(
                [x0 + 2 * width / 5, y0], width / 5, height, facecolor=self.edgecolor,
                transform=handlebox.get_transform())
            handlebox.add_artist(patch1)

        return patch

class Stats:

    def __init__(self, folder):
        self.folder = folder
        self.filez = os.listdir(folder)

    def evaluate_file(self, path):
        with open(path) as file:
            data = json.load(file)
        results = {}
        results["samples"] = len(data["local_list"])
        lock_speedups = list(map(lambda x, y: x / y, data["seq"], data["mutex"]))
        results["lock_speedup"] = sum(lock_speedups) / len(lock_speedups)
        var_lock = list(map(lambda x: (x - results["lock_speedup"])**2, lock_speedups))
        results["lock_var"] = (1 / (results["samples"] - 1)) * sum(var_lock)
        local_list = list(map(lambda x, y: x / y, data["seq"], data["local_list"]))
        results["list_speedup"] = sum(local_list) / len(local_list)
        var_list = list(map(lambda x: (x - results["list_speedup"]) ** 2, local_list))
        results["list_var"] = (1 / (results["samples"] - 1)) * sum(var_list)
        return results

    def evaluate_single_axis(self, axis):
        vals = set(map(lambda x: x.split("_")[axis], self.filez))
        try:
            vals = (list(map(lambda x: float(x.split(".")[0]), vals)))
        except Exception as e:
            vals = (list(map(lambda x: float(x), vals)))
        vals.sort()
        results = {}
        results["lock_speedup"] = []
        results["list_speedup"] = []
        results["lock_var"] = []
        results["list_var"] = []
        results["x_axis"] = vals
        for val in vals:
            val_filez = list(filter(lambda x: float(x.split("_")[axis].split(".")[0]) == val, self.filez))
            curr_results = {}
            curr_results["lock_speedup"] = []
            curr_results["list_speedup"] = []
            curr_results["lock_var"] = []
            curr_results["list_var"] = []
            for file in val_filez:
                current = self.evaluate_file(self.folder + os.sep + file)
                curr_results["lock_speedup"].append(current["lock_speedup"])
                curr_results["list_var"].append(current["list_var"])
                curr_results["list_speedup"].append(current["list_speedup"])
                curr_results["lock_var"].append(current["lock_var"])
            curr_results["lock_speedup"] = sum(curr_results["lock_speedup"]) / len(curr_results["lock_speedup"])
            curr_results["list_var"] = sum(curr_results["list_var"]) / len(curr_results["list_var"])
            curr_results["list_speedup"] = sum(curr_results["list_speedup"]) / len(curr_results["list_speedup"])
            curr_results["lock_var"] = sum(curr_results["lock_var"]) / len(curr_results["lock_var"])
            results["list_speedup"].append(curr_results["list_speedup"])
            results["list_var"].append(curr_results["list_var"])
            results["lock_var"].append(curr_results["lock_var"])
            results["lock_speedup"].append(curr_results["lock_speedup"])
        return results

    def plot_mean_and_CI(self, x, means, lb, ub, color_mean=None, color_shading=None):
        # plot the shaded range of the confidence intervals
        plt.fill_between(x, ub, lb,
                         color=color_shading, alpha=.5)
        # plot the mean on top
        plt.plot(x, means, color_mean)

    def plot_axis(self, axis):
        axis_data = self.evaluate_single_axis(axis)
        # acquire mutex data
        mut_speed = axis_data["lock_speedup"]
        two_std = list(map(lambda x: 2 * sqrt(x), axis_data["lock_var"]))
        mut_upper_bound = list(map(lambda x, y: x + y, mut_speed, two_std))
        mut_lower_bound = list(map(lambda x, y: x - y, mut_speed, two_std))
        mut_speed = np.array(mut_speed)
        mut_upper_bound = np.array(mut_upper_bound)
        mut_lower_bound = np.array(mut_lower_bound)
        # acquire local list data
        list_speed = axis_data["list_speedup"]
        two_std = list(map(lambda x: 2 * sqrt(x), axis_data["list_var"]))
        list_upper_bound = list(map(lambda x, y: x + y, list_speed, two_std))
        list_lower_bound = list(map(lambda x, y: x - y, list_speed, two_std))
        list_speed = np.array(list_speed)
        list_upper_bound = np.array(list_upper_bound)
        list_lower_bound = np.array(list_lower_bound)
        fig = plt.figure(1, figsize=(7, 2.5))
        x_axis = np.array(axis_data["x_axis"])
        self.plot_mean_and_CI(x_axis, mut_speed, mut_upper_bound, mut_lower_bound, color_mean='k', color_shading='k')
        self.plot_mean_and_CI(x_axis, list_speed, list_upper_bound, list_lower_bound, color_mean='r--', color_shading='r')
        bg = np.array([1, 1, 1])  # background of the legend is white
        colors = ['black', 'red']
        # with alpha = .5, the faded color is the average of the background and color
        colors_faded = [(np.array(cc.to_rgb(color)) + bg) / 2.0 for color in colors]
        plt.xlabel('threads')
        plt.ylabel('speed up')
        plt.legend([0, 1], ['Lock', 'Local List'],
                   handler_map={
                       0: LegendObject(colors[0], colors_faded[0]),
                       1: LegendObject(colors[1], colors_faded[1], dashed=True),
                   })
        plt.title('aggregated speed up')
        #plt.tight_layout()
        plt.grid()
        plt.show()

    def evaluate_two_axis_algo(self, axis1, axis2, algo):
        axis1_vals = set(map(lambda x: x.split("_")[axis1], self.filez))
        try:
            axis1_vals = (list(map(lambda x: float(x), axis1_vals)))
        except Exception as e:
            axis1_vals = (list(map(lambda x: float(x.split(".")[0]), axis1_vals)))
        axis1_vals.sort()
        axis2_vals = set(map(lambda x: x.split("_")[axis2], self.filez))
        try:
            axis2_vals = (list(map(lambda x: float(x), axis2_vals)))
        except Exception as e:
            axis2_vals = (list(map(lambda x: float(x.split(".")[0]), axis2_vals)))
        axis2_vals.sort()
        speed_ups = np.zeros((len(axis1_vals), len(axis2_vals)))
        vars = np.zeros((len(axis1_vals), len(axis2_vals)))
        for i in range(0, len(axis1_vals)):
            for j in range(0, len(axis2_vals)):
                curr_x = axis1_vals[i]
                curr_y = axis2_vals[j]
                if axis1 != MATCH_PERC:
                    val_filez = list(filter(lambda x: float(x.split("_")[axis1].split(".")[0]) == curr_x, self.filez))
                else:
                    val_filez = list(filter(lambda x: float(x.split("_")[axis1]) == curr_x, self.filez))
                if axis2 != MATCH_PERC:
                    val_filez = list(filter(lambda x: float(x.split("_")[axis2].split(".")[0]) == curr_y, val_filez))
                else:
                    val_filez = list(filter(lambda x: float(x.split("_")[axis2]) == curr_y, val_filez))
                cur_speed_ups = []
                cur_vars = []
                for file in val_filez:
                    current = self.evaluate_file(self.folder + os.sep + file)
                    cur_speed_ups.append(current[algo + "_speedup"])
                    cur_vars.append(current[algo + "_var"])
                speed_ups[i][j] = sum(cur_speed_ups) / len(cur_speed_ups)
                vars[i][j] = sum(cur_vars) / len(cur_vars)
        return speed_ups, vars

    def plot_scale_two(self, algo):
        speed_ups, vars = self.evaluate_two_axis_algo(THREADS, MATCH_PERC, algo)
        axis1_vals = set(map(lambda x: x.split("_")[THREADS], self.filez))
        try:
            axis1_vals = (list(map(lambda x: float(x), axis1_vals)))
        except Exception as e:
            axis1_vals = (list(map(lambda x: float(x.split(".")[0]), axis1_vals)))
        axis1_vals.sort()
        axis2_vals = set(map(lambda x: x.split("_")[MATCH_PERC], self.filez))
        try:
            axis2_vals = (list(map(lambda x: float(x), axis2_vals)))
        except Exception as e:
            axis2_vals = (list(map(lambda x: float(x.split(".")[0]), axis2_vals)))
        axis2_vals.sort()
        axis1_vals = list(map(lambda x: str(int(x)), axis1_vals))
        axis2_vals = list(map(lambda x: str(int(round(x * 100))), axis2_vals))
        if algo == "list":
            name = "Local list"
        else:
            name = "Lock"
        plot_heat_map(speed_ups, axis1_vals, axis2_vals, "match set size (in percent)", "threads", "speed up", name + " speed up", SPEED_UP_BORDERS)
        plot_heat_map(vars, axis1_vals, axis2_vals, "match set size (in percent)", "threads", "variance", name + " variance", VAR_BORDERS)

    def plot_problemsize(self, algo):
        old_filez = self.filez
        self.filez = list(filter(lambda x: "12" in x.split("_")[THREADS], self.filez))
        speed_ups, vars = self.evaluate_two_axis_algo(SIZE, DIM, algo)
        axis1_vals = set(map(lambda x: x.split("_")[SIZE], self.filez))
        try:
            axis1_vals = (list(map(lambda x: float(x), axis1_vals)))
        except Exception as e:
            axis1_vals = (list(map(lambda x: float(x.split(".")[0]), axis1_vals)))
        axis1_vals.sort()
        axis2_vals = set(map(lambda x: x.split("_")[DIM], self.filez))
        try:
            axis2_vals = (list(map(lambda x: float(x), axis2_vals)))
        except Exception as e:
            axis2_vals = (list(map(lambda x: float(x.split(".")[0]), axis2_vals)))
        axis2_vals.sort()
        axis1_vals = list(map(lambda x: str(int(x)), axis1_vals))
        axis2_vals = list(map(lambda x: str(int(round(x))), axis2_vals))
        if algo == "list":
            name = "Local list"
        else:
            name = "Lock"
        plot_heat_map(speed_ups.transpose(), axis2_vals, axis1_vals, "population size", "dimension", "speed up", name + " speed up", (2, 3.5))
        plot_heat_map(vars.transpose(), axis2_vals, axis1_vals, "population size", "dimension", "variance", name + " variance", (0, 0.25))
        self.filez = old_filez

    def plot_speed_diff_size_p_vals(self):
        old_filez = self.filez
        self.filez = list(filter(lambda x: "12" in x.split("_")[THREADS], self.filez))
        list_speed_ups, list_vars = self.evaluate_two_axis_algo(SIZE, DIM, LIST_ALGO)
        lock_speed_ups, lock_vars = self.evaluate_two_axis_algo(SIZE, DIM, LOCK_ALGO)
        speed_up_diff = list_speed_ups - lock_speed_ups
        vars = list_vars + lock_vars
        samples = self.evaluate_file(self.folder + os.sep + self.filez[0])["samples"]
        vars = np.sqrt(((samples - 1) * vars) / (2 * samples - 2))
        test_stats = sqrt(samples / 2) * (speed_up_diff / vars)
        p_vals = np.zeros(test_stats.shape)
        for i in range(0, vars.shape[0]):
            for j in range(0, vars.shape[1]):
                p_vals[i][j] = 1 - t.cdf(test_stats[i][j], df=2 * samples - 2)

        axis1_vals = set(map(lambda x: x.split("_")[SIZE], self.filez))
        try:
            axis1_vals = (list(map(lambda x: float(x), axis1_vals)))
        except Exception as e:
            axis1_vals = (list(map(lambda x: float(x.split(".")[0]), axis1_vals)))
        axis1_vals.sort()
        axis2_vals = set(map(lambda x: x.split("_")[DIM], self.filez))
        try:
            axis2_vals = (list(map(lambda x: float(x), axis2_vals)))
        except Exception as e:
            axis2_vals = (list(map(lambda x: float(x.split(".")[0]), axis2_vals)))
        axis2_vals.sort()
        axis1_vals = list(map(lambda x: str(int(x)), axis1_vals))
        axis2_vals = list(map(lambda x: str(int(round(x))), axis2_vals))
        fig, ax = plt.subplots()
        im, cbar = heatmap(p_vals, axis1_vals, axis2_vals, ax=ax,
                           cmap="YlGn", cbarlabel="p values")
        self.annotate_heatmap_sig(im)
        plt.xlabel("population size")
        plt.ylabel("dimension")
        plt.title("p values for the speed up differences")
        plt.show()
        self.filez = old_filez

    def plot_speed_diff_size(self):
        old_filez = self.filez
        self.filez = list(filter(lambda x: "12" in x.split("_")[THREADS], self.filez))
        list_speed_ups, list_vars = self.evaluate_two_axis_algo(SIZE, DIM, LIST_ALGO)
        lock_speed_ups, lock_vars = self.evaluate_two_axis_algo(SIZE, DIM, LOCK_ALGO)
        speed_up_diff = list_speed_ups - lock_speed_ups
        vars = list_vars + lock_vars
        axis1_vals = set(map(lambda x: x.split("_")[SIZE], self.filez))
        try:
            axis1_vals = (list(map(lambda x: float(x), axis1_vals)))
        except Exception as e:
            axis1_vals = (list(map(lambda x: float(x.split(".")[0]), axis1_vals)))
        axis1_vals.sort()
        axis2_vals = set(map(lambda x: x.split("_")[DIM], self.filez))
        try:
            axis2_vals = (list(map(lambda x: float(x), axis2_vals)))
        except Exception as e:
            axis2_vals = (list(map(lambda x: float(x.split(".")[0]), axis2_vals)))
        axis2_vals.sort()
        axis1_vals = list(map(lambda x: str(int(x)), axis1_vals))
        axis2_vals = list(map(lambda x: str(int(round(x))), axis2_vals))
        plot_surface(speed_up_diff, axis1_vals, axis2_vals, "dimension", "population size", "speed up (local list - lock)", "Difference in speed up")
        plot_heat_map(vars.transpose(), axis2_vals, axis1_vals, "population size", "dimension", "variance", "Speed up difference variance", (0, 0.25))
        self.filez = old_filez

    def t_test_speed_diff(self):
        samples = self.evaluate_file(self.folder + os.sep + self.filez[0])["samples"]
        list_speed_ups, list_vars = self.evaluate_two_axis_algo(THREADS, MATCH_PERC, LIST_ALGO)
        lock_speed_ups, lock_vars = self.evaluate_two_axis_algo(THREADS, MATCH_PERC, LOCK_ALGO)
        speed_up_diff = list_speed_ups - lock_speed_ups
        vars = list_vars + lock_vars
        vars = np.sqrt(((samples - 1) * vars) / (2 * samples - 2))
        test_stats = sqrt(samples / 2) * (speed_up_diff / vars)
        p_vals = np.zeros(test_stats.shape)
        for i in range(0, vars.shape[0]):
            for j in range(0, vars.shape[1]):
                p_vals[i][j] = 1 - t.cdf(test_stats[i][j], df=2 * samples - 2)
        return p_vals

    def annotate_heatmap_sig(self, im, data=None, valfmt="{x:.2f}",
                         textcolors=["black", "white"],
                         threshold=None, **textkw):
        """
        A function to annotate a heatmap.

        Parameters
        ----------
        im
            The AxesImage to be labeled.
        data
            Data used to annotate.  If None, the image's data is used.  Optional.
        valfmt
            The format of the annotations inside the heatmap.  This should either
            use the string format method, e.g. "$ {x:.2f}", or be a
            `matplotlib.ticker.Formatter`.  Optional.
        textcolors
            A list or array of two color specifications.  The first is used for
            values below a threshold, the second for those above.  Optional.
        threshold
            Value in data units according to which the colors from textcolors are
            applied.  If None (the default) uses the middle of the colormap as
            separation.  Optional.
        **kwargs
            All other arguments are forwarded to each call to `text` used to create
            the text labels.
        """

        if not isinstance(data, (list, np.ndarray)):
            data = im.get_array()

        # Normalize the threshold to the images color range.
        if threshold is not None:
            threshold = im.norm(threshold)
        else:
            threshold = im.norm(data.max()) / 2.

        # Set default alignment to center, but allow it to be
        # overwritten by textkw.
        kw = dict(horizontalalignment="center",
                  verticalalignment="center")
        kw.update(textkw)

        # Get the formatter in case a string is supplied
        if isinstance(valfmt, str):
            valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        texts = []
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i][j] < 0.05:
                    kw.update(color=textcolors[int(im.norm(data[i, j]) < threshold)])
                    # kw.update(color=textcolors["white"])
                    # text = im.axes.text(j, i, valfmt(0.1, None), **kw)
                    text = im.axes.text(j, i, "*", **kw)
                    texts.append(text)
        return texts

    def plot_p_vals_diff(self):
        p_vals = self.t_test_speed_diff()
        axis1_vals = set(map(lambda x: x.split("_")[THREADS], self.filez))
        try:
            axis1_vals = (list(map(lambda x: float(x), axis1_vals)))
        except Exception as e:
            axis1_vals = (list(map(lambda x: float(x.split(".")[0]), axis1_vals)))
        axis1_vals.sort()
        axis2_vals = set(map(lambda x: x.split("_")[MATCH_PERC], self.filez))
        try:
            axis2_vals = (list(map(lambda x: float(x), axis2_vals)))
        except Exception as e:
            axis2_vals = (list(map(lambda x: float(x.split(".")[0]), axis2_vals)))
        axis2_vals.sort()
        axis1_vals = list(map(lambda x: str(int(x)), axis1_vals))
        axis2_vals = list(map(lambda x: str(int(round(x * 100))), axis2_vals))
        fig, ax = plt.subplots()
        im, cbar = heatmap(p_vals, axis1_vals, axis2_vals, ax=ax,
                           cmap="YlGn", cbarlabel="p values")
        self.annotate_heatmap_sig(im)
        plt.xlabel("match set size (in percent)")
        plt.ylabel("threads")
        plt.title("p values for the speed up differences")
        plt.show()

    def plot_speed_diff(self):
        list_speed_ups, list_vars = self.evaluate_two_axis_algo(THREADS, MATCH_PERC, LIST_ALGO)
        lock_speed_ups, lock_vars = self.evaluate_two_axis_algo(THREADS, MATCH_PERC, LOCK_ALGO)
        speed_up_diff = list_speed_ups - lock_speed_ups
        vars = list_vars + lock_vars
        axis1_vals = set(map(lambda x: x.split("_")[THREADS], self.filez))
        try:
            axis1_vals = (list(map(lambda x: float(x), axis1_vals)))
        except Exception as e:
            axis1_vals = (list(map(lambda x: float(x.split(".")[0]), axis1_vals)))
        axis1_vals.sort()
        axis2_vals = set(map(lambda x: x.split("_")[MATCH_PERC], self.filez))
        try:
            axis2_vals = (list(map(lambda x: float(x), axis2_vals)))
        except Exception as e:
            axis2_vals = (list(map(lambda x: float(x.split(".")[0]), axis2_vals)))
        axis2_vals.sort()
        axis1_vals = list(map(lambda x: str(int(x)), axis1_vals))
        axis2_vals = list(map(lambda x: str(int(round(x * 100))), axis2_vals))
        plot_surface(speed_up_diff, axis1_vals, axis2_vals, "match set size (in percent)", "threads", "speed up (local list - lock)", "Difference in speed up")
        plot_heat_map(vars, axis1_vals, axis2_vals, "match set size (in percent)", "threads", "variance", "Speed up difference variance")

def plot_surface(data, x_axis, y_axis, x_axis_name, y_axis_name, label, title):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Make data
    X = np.array(list(map(lambda x: int(x), x_axis)))
    Y = np.array(list(map(lambda x: int(x), y_axis)))
    X, Y = np.meshgrid(X, Y)
    Z = data.transpose()
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.ylabel(x_axis_name, labelpad=40)
    plt.xlabel(y_axis_name, labelpad=40)
    plt.tick_params(axis='x', rotation=-45)
    plt.tick_params(axis='y')
    plt.tick_params(axis ='z', rotation=45)
    plt.title(title, pad=20)
    # Customize the z axis.
    #ax.set_zlim(-1.01, 1.01)
    #ax.zaxis.set_major_locator(LinearLocator(10))
    #ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_heat_map(data, x_axis, y_axis, x_axis_name, y_axis_name, label, title, borders):
    fig, ax = plt.subplots()
    im, cbar = heatmap(data, x_axis, y_axis, borders, ax=ax,
                       cmap="YlGn", cbarlabel=label)
    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.title(title)
    plt.tight_layout()
    plt.subplots_adjust(top=0.78)
    #plt.clim(1, 3.5)
    plt.show()


#stat = Stats(r"..\results")
#stat.plot_speed_diff_size()
#stat.plot_problemsize(LOCK_ALGO)
#stat.plot_axis(THREADS)
#stat.plot_scale_two(LOCK_ALGO)
#stat.plot_speed_diff_size()
#plot_speed_diff_size_p_vals()
#stat.plot_p_vals_diff()
# stat.plot_speed_diff_size()
#stat.plot_speed_diff()
#stat.plot_scale_two(LIST_ALGO)
#stat.plot_scale_two(LOCK_ALGO)
'''
speed_ups, vars = stat.evaluate_two_axis_algo(THREADS, MATCH_PERC, LOCK_ALGO)
ax = sns.heatmap(speed_ups, linewidth=0.5)
plt.xlabel('size match set')
plt.ylabel('threads')
plt.title('aggregated speed up')
plt.show()
'''
# print(speed_ups)
# print("----------")
# print(vars)
#stat.plot_axis(5)
#stat.evaluate_file(r"..\results\10_10000_3_100_0.1_2.json")
# print(stat.evaluate_single_axis(5))
