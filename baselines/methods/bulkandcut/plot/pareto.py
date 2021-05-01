import os
import csv
import shutil
import math
from typing import Tuple, List
from collections import namedtuple
from datetime import datetime
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import PIL

from baselines.methods.bulkandcut import rng


Benchmark = namedtuple("Benchmark", ["name", "data", "plot_front", "marker", "color"])

fig_h = 6.2  # 6.2 inches - the default Libre-office slide height
fig_w = fig_h * 16. / 9.  # widescreen aspect ratio (16:9)


def generate_pareto_animation(working_dir: str,
                              ref_point: Tuple[float],
                              benchmarks: List[Benchmark] = None,
                              ):
    # Create output directory
    figures_dir = os.path.join(working_dir, "pareto")
    if os.path.exists(figures_dir):
        shutil.rmtree(figures_dir)
    os.makedirs(figures_dir)

    population = _load_csv(working_dir=working_dir)
    pop_size = len(population)
    hyper_volumes = []
    first_bulkup, first_slimdown = _phase_transitions(population)
    start_time = population[0]["birth"]
    the_time = {
        "now": 0,
        "bulkup": (population[first_bulkup]["birth"] - start_time).seconds / 3600.,
        "slimdown": (population[first_slimdown]["birth"] - start_time).seconds / 3600.,
        "end": (population[-1]["birth"] - start_time).seconds / 3600.,
        }

    for i in range(pop_size + 1):
        print(f"Generating frame {i} of {pop_size}")
        frame_path = os.path.join(figures_dir, str(i).rjust(4, "0") + ".png")
        the_time["now"] = (population[max(0, i-1)]["birth"] - start_time).seconds / 3600.
        sub_population = population[:i]
        pareto_front, dominated_set = _pareto_front(population=sub_population)
        hyper_vol = _hyper_volume_2D(pareto_front, ref_point)
        arrow = _connector(population=sub_population)
        _render_a_frame(
            title=_title_string(sub_population, hyper_vol),
            pareto_front=pareto_front,
            ref_point=ref_point,
            dominated_set=dominated_set,
            arrow=arrow,
            frame_path=frame_path,
            the_time=the_time,
            benchmarks=benchmarks,
            )
        hyper_volumes.append(hyper_vol)

    print("Generating GIF")
    _generate_gif(figs_dir=figures_dir)

    print("Generating hyper-volume vs time plot")
    _plot_volume_vs_training_time(
        population=population,
        hyper_volumes=hyper_volumes,
        first_bulkup=first_bulkup,
        first_slimdown=first_slimdown,
        fig_dir=figures_dir,
        )


def _load_csv(working_dir):
    query = os.path.join(working_dir, "population_summary.csv")
    csv_path = glob(query)[0]
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        csv_content = []
        for row in reader:
            csv_content.append({
                "n_pars": int(row["n_parameters"]),
                "neg_acc": -float(row["accuracy"]) + rng.uniform() * 1E-8,  # Add a tiebreaker noise
                "parent": int(row["parent_id"]),
                "n_bulks": int(row["bulk_counter"]),
                "n_cuts": int(row["cut_counter"]),
                "birth": datetime.strptime(row["birth"], "%Y-%m-%d %H:%M:%S.%f"),
            })
    return csv_content


def _phase_transitions(population):
    first_bulkup, first_slimdown = -1, -1
    for n, ind in enumerate(population):
        if first_bulkup == -1 and ind["n_bulks"] > 0:
            first_bulkup = n
        if ind["n_cuts"] > 0:
            first_slimdown = n
            break
    return first_bulkup, first_slimdown


def _hyper_volume_2D(pareto_front, ref_point):
    ref_x = math.log10(ref_point[0])
    x_segments = np.log10(pareto_front[:, 0])
    x_segments = np.clip(x_segments, a_min=None, a_max=ref_x)  # Exclude invalid volumes
    x_segments = ref_x - x_segments

    y_segments = np.clip(pareto_front[:, 1], a_min=None, a_max=ref_point[1])
    y_segments = ref_point[1] - y_segments
    y_segments[1:] -= y_segments[:-1]

    hyper_vol = np.sum(x_segments * y_segments)
    return hyper_vol


def _individual_cost(population, indv_id=-1):
    indv = population[indv_id]
    n_pars = int(indv["n_pars"])
    neg_acc = float(indv["neg_acc"])
    return np.array([n_pars, neg_acc])


def _pareto_front(population):
    num_of_pars = np.array([ind["n_pars"] for ind in population])[:, np.newaxis]
    neg_accuracy = np.array([ind["neg_acc"] for ind in population])[:, np.newaxis]
    costs = np.hstack((num_of_pars, neg_accuracy))
    not_eye = np.logical_not(np.eye(len(costs)))  # False in the main diagonal, True elsew.

    worst_at_num_pars = np.less_equal(num_of_pars, num_of_pars.T)
    worst_at_accuracy = np.less_equal(neg_accuracy, neg_accuracy.T)
    worst_at_both = np.logical_and(worst_at_num_pars, worst_at_accuracy)
    worst_at_both = np.logical_and(worst_at_both, not_eye)  # excludes self-comparisons
    domination = np.any(worst_at_both, axis=0)

    dominated_set = costs[domination]
    pareto_front = costs[np.logical_not(domination)]
    pareto_front = pareto_front[np.argsort(pareto_front[:, 0])]  # sort by x (n_pars)
    return pareto_front, dominated_set


def _pareto_front_coords(pareto_front, ref_point):
    pareto_coords = []
    for i in range(len(pareto_front) - 1):
        pareto_coords.append(pareto_front[i])
        y1 = pareto_front[i][1]
        x2 = pareto_front[i + 1][0]
        pareto_coords.append([x2, y1])
    pareto_coords.append(pareto_front[-1])
    pareto_coords.append([ref_point[0], pareto_front[-1][1]])

    dominated_area = list(pareto_coords)
    dominated_area.append([ref_point[0], pareto_coords[-1][1]])
    dominated_area.append([ref_point[0], ref_point[1]])
    dominated_area.append([pareto_coords[0][0], ref_point[1]])

    return np.array(pareto_coords), np.array(dominated_area)


def _connector(population):
    if len(population) < 2:
        return None
    parent_id = population[-1]["parent"]
    if parent_id == -1:
        return None
    child_nbulk = population[-1]["n_bulks"]
    parent_nbulk = population[parent_id]["n_bulks"]
    arrow_type = "bulk" if child_nbulk > parent_nbulk else "cut"

    child_cost = _individual_cost(population=population)
    parent_cost = _individual_cost(population=population, indv_id=parent_id)
    coords = np.vstack((child_cost, parent_cost))

    return arrow_type, coords


def _title_string(sub_population, dominated_area):
    if len(sub_population) < 1:
        return ""
    title = f"Hyper volume: {dominated_area:.2f}\n"
    ind_id = len(sub_population) - 1
    parent_id = sub_population[-1]["parent"]
    title += "Newcomer:" + str(ind_id).rjust(4, "0") + "\n"
    if parent_id != -1:
        title += "(an offspring of " + str(parent_id).rjust(4, "0") + ")"
    return title


def _render_a_frame(title: str,
                    frame_path: str,
                    ref_point: Tuple[float],
                    benchmarks: List[Benchmark] = [],
                    pareto_front: "np.array" = np.array([]),
                    dominated_set: "np.array" = np.array([]),
                    the_time: dict = None,
                    arrow: tuple = None,
                    ):
    # Global figure settings:
    n_rows = 10
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#f6f6f6")
    fig.suptitle(title, fontdict={"family": "monospace"})

    # x-axis
    plt.subplot(n_rows, 1, (1, n_rows - 2))
    plt.xscale("log")
    plt.xlabel("number of parameters")
    x_max_exp = int(math.log10(ref_point[0]))
    plt.xlim((.9, ref_point[0] + 10 ** (x_max_exp - 1)))
    plt.xticks(ticks=[10. ** n for n in range(x_max_exp + 1)])

    # y-axis:
    plt.ylabel("negative accuracy (%)")
    plt.ylim((-100., ref_point[1] + 1.))
    plt.yticks(ticks=range(0, -101, -10))

    # Reference point
    plt.scatter(
        x=ref_point[0],
        y=ref_point[1],
        s=30.,
        marker="s",
        color="k",
        label="ref point",
        )

    # Benchmarks
    for bch in benchmarks:
        plt.scatter(
            x=bch.data[:, 0],
            y=bch.data[:, 1],
            s=30.,
            marker=bch.marker,
            color=bch.color,
            label=bch.name,
            zorder=100,  # Make sure the benchmarks are always visible
            )
        if bch.plot_front:
            bch_front, _ = _pareto_front_coords(bch.data, ref_point)
            plt.plot(
                bch_front[:-1, 0],
                bch_front[:-1, 1],
                color=bch.color,
                zorder=100,
                )

    # Dominated solutions:
    if dominated_set is not None and len(dominated_set) > 0:
        plt.scatter(
            x=dominated_set[:, 0],
            y=dominated_set[:, 1],
            marker=".",
            s=60.,
            alpha=.6,
            color="tab:grey",
            )

    # Pareto-optimal solutions:
    p_col = "tab:red"
    if len(pareto_front) > 0:
        front_coords, dominated_area = _pareto_front_coords(pareto_front, ref_point)
        plt.scatter(
            x=pareto_front[:, 0],
            y=pareto_front[:, 1],
            marker="*",
            color=p_col,
            label="bulk and cut",
            )
        plt.plot(front_coords[:, 0], front_coords[:, 1], alpha=.5, color=p_col)
        plt.fill(dominated_area[:, 0], dominated_area[:, 1], alpha=.2, color=p_col)

    # Parent-to-child connection:
    if arrow is not None:
        ar_type = arrow[0]
        ar_coords = arrow[1]
        plt.plot(
            ar_coords[:, 0],
            ar_coords[:, 1],
            color="m" if ar_type == "bulk" else "c",
            )

    # Add legend box
    plt.legend(loc="lower left")

    # Time line
    plt.subplot(n_rows, 1, n_rows)
    plt.xlim((0., the_time["end"]))
    plt.xlabel("hours")
    plt.yticks([0], ["Time"])
    plt.grid(b=None)
    plt.barh(
        y=0,
        width=the_time["now"],
        alpha=.3,
        color="tab:red",
    )
    plt.axvline(the_time["bulkup"], c="tab:gray", linestyle="--")
    plt.axvline(the_time["slimdown"], c="tab:gray", linestyle="--")

    # Save figure
    fig.savefig(frame_path)
    plt.close(fig)


def _generate_gif(figs_dir, sampling: int = 1, scale: float = 1.):
    imgs = []
    query = os.path.join(figs_dir, "*.png")
    fig_paths = sorted(glob(query))[::sampling]
    for fpath in fig_paths:
        img = PIL.Image.open(fpath)
        new_w = int(scale * img.size[0])
        new_h = int(scale * img.size[1])
        img = img.resize((new_w, new_h))
        imgs.append(img.copy())  # Workaround to avoid the "too many files open" exception
        img.close()
    gif_path = os.path.join(figs_dir, "animated_pareto_front.gif")
    imgs[0].save(gif_path, save_all=True, append_images=imgs[1:], loop=0, duration=40.)


def _plot_volume_vs_training_time(population, hyper_volumes, first_bulkup, first_slimdown, fig_dir):
    plt.style.use("ggplot")
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#f6f6f6")
    plt.plot(hyper_volumes, color="tab:red")
    plt.xlabel("Individual id")
    plt.ylabel("Hyper-volume")
    plt.ylim((300, None))
    plt.axvline(first_bulkup, color="tab:purple", label="Bulk-up begin", linestyle="--")
    plt.axvline(first_slimdown, color="tab:green", label="Slim-down begin", linestyle="--")
    plt.legend()
    fig.savefig(os.path.join(fig_dir, "volumes.png"))
    plt.close(fig)
