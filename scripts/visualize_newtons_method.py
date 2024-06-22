""""Visualizing Newton's method in 1D."""

from pathlib import Path
from typing import Callable, List

import matplotlib
import numpy as np
from jax import grad, hessian
from jax import numpy as jnp
from matplotlib import animation
from matplotlib import pyplot as plt


def f(x: float) -> float:
    """A function we want to optimize."""
    # This example is tricky because Hessian may be not PSD. Need to modify
    # Hessian to handle, or just initialize carefully (as done below).
    return -3 + 0.001 * (
        -((x - 3) ** 2)
        - 2 * x
        - 5
        + jnp.sin(x) * (x + 1) ** 3
        - jnp.cos(x) * (x - 2) ** 4
    )


def _create_taylor_expansion(x0: float) -> Callable[[float], float]:
    f_x0 = f(x0)
    grad_f_x0 = grad(f)(x0)
    hess_f_x0 = hessian(f)(x0)

    def _taylor_expansion(x0_plus_t: float) -> float:
        t = x0_plus_t - x0
        return f_x0 + grad_f_x0 * t + 0.5 * hess_f_x0 * t**2

    return _taylor_expansion


def _main(outdir: Path, fps: int, num_iters: int) -> None:

    xs = np.linspace(-10, 10, num=1000)
    x0 = -8.0
    taylor_expansion = _create_taylor_expansion(x0)
    x_is = [x0]
    x_i = x0

    for _ in range(num_iters + 1):
        x_i = x_i - grad(f)(x_i) / hessian(f)(x_i)
        x_is.append(float(x_i))

    matplotlib.rcParams.update({"font.size": 20})
    fig = plt.figure(figsize=(10, 6))
    plt.plot(xs, f(xs), label="f(x)", color="black")
    (taylor_line,) = plt.plot(
        xs, taylor_expansion(xs), label="Taylor", color="blue", linestyle="--"
    )
    dot_line = plt.scatter(
        [x0], [f(x0)], label="Current x", color="red", marker="o", s=100
    )
    next_x_line = plt.vlines(
        x_is[1], -10, 10, label="Next x", color="gray", linestyle="--"
    )
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-10, 0)
    plt.ylim(-10, 10)
    plt.legend()
    plt.xticks([])
    plt.yticks([])
    plt.title("Newton's Method")
    plt.tight_layout()

    def _update_plt(i: int) -> List[plt.Line2D]:
        x_i = x_is[i]
        next_x_i = x_is[i + 1]
        taylor_expansion = _create_taylor_expansion(x_i)
        dot_line.set_offsets([x_i, f(x_i)])
        taylor_line.set_ydata(taylor_expansion(xs))
        next_x_line.set_segments([[[next_x_i, -10], [next_x_i, 10]]])
        return [dot_line, taylor_line, next_x_line]

    frames = num_iters
    interval = 1000 / fps
    ani = animation.FuncAnimation(
        fig=fig, func=_update_plt, frames=frames, interval=interval
    )
    outfile = outdir / "newtons_method.mp4"
    ani.save(outfile)
    print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=Path("results"), type=Path)
    parser.add_argument("--fps", default=2, type=int)
    parser.add_argument("--num_iters", default=10, type=int)
    parser_args = parser.parse_args()
    _main(
        parser_args.outdir,
        parser_args.fps,
        parser_args.num_iters,
    )
