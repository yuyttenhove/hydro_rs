from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from make_ics_comparison import box, triangle, wave, transform


def godunov(ics: np.ndarray, t_end: float, a: float = 1., delta_x: float = 0.01, cfl: float = 0.8) -> np.ndarray:
    t = 0
    delta_t = cfl * delta_x / abs(a)
    c = delta_t / delta_x
    sol = np.array(ics)
    for _ in tqdm(np.arange(0, t_end, delta_t)):
        if a > 0.:
            fluxes = a * sol
        else:
            fluxes = a * np.roll(sol, -1)
        sol -= c * fluxes
        sol += c * np.roll(fluxes, 1)
        t += delta_t
    return sol


def slope_limiter_none(slopes, _differences):
    return slopes


def slope_limiter_direct(slopes, differences, beta=2.):
    limited = np.zeros_like(slopes)
    differences_rolled = np.roll(differences, 1)
    pos = differences > 0
    limited[pos] = np.maximum(0, np.maximum(np.minimum(beta * differences_rolled[pos], differences[pos]),
                                            np.minimum(differences_rolled[pos], beta * differences[pos])))
    limited[~pos] = np.minimum(0, np.minimum(np.maximum(beta * differences_rolled[~pos], differences[~pos]),
                                             np.maximum(differences_rolled[~pos], beta * differences[~pos])))
    return limited


def muscl_hancock(ics: np.ndarray, t_end: float, a: float = 1., delta_x: float = 0.01, cfl: float = 0.8,
                  limiter: Callable[[np.ndarray, np.ndarray], np.ndarray] = slope_limiter_direct) -> np.ndarray:
    t = 0
    delta_t = cfl * delta_x / abs(a)
    c = delta_t / delta_x
    sol = np.array(ics)
    for _ in tqdm(np.arange(0, t_end, delta_t)):
        differences = np.roll(sol, -1) - sol
        slopes = 0.5 * (differences + np.roll(differences, 1))
        limited = limiter(slopes, differences)
        uR = sol + 0.5 * limited
        uL = sol - 0.5 * limited
        time_extrapolations = 0.5 * c * a * (uL - uR)
        if a > 0.:
            fluxes = a * (uR + time_extrapolations)
        else:
            fluxes = a * np.roll(uL + time_extrapolations, -1)
        sol -= c * fluxes
        sol += c * np.roll(fluxes, 1)
        t += delta_t
    return sol


def flux_limiter_none(flow_parameter: np.ndarray) -> np.ndarray:
    return np.ones_like(flow_parameter)


def flux_limiter_vanleer(flow_parameter: np.ndarray) -> np.ndarray:
    flow_parameter_abs = np.abs(flow_parameter)
    return np.maximum(0., (flow_parameter + flow_parameter_abs) / (1 + flow_parameter_abs))


def flux_limiter_mc(flow_parameter: np.ndarray) -> np.ndarray:
    return np.maximum(0., np.minimum(np.minimum(0.5 * (1. + flow_parameter), 2. * flow_parameter), 2.))


def flux_limiter_minbee(flow_parameter: np.ndarray) -> np.ndarray:
    return np.maximum(0., np.minimum(1., flow_parameter))


def flux_limiter_superbee(flow_parameter: np.ndarray) -> np.ndarray:
    return np.maximum(0., np.maximum(np.minimum(1., 2. * flow_parameter), np.minimum(2., flow_parameter)))


def waf(ics: np.ndarray, t_end: float, a: float = 1., delta_x: float = 0.01, cfl: float = 0.8,
        limiter: Callable[[np.ndarray], np.ndarray] = flux_limiter_vanleer) -> np.ndarray:
    t = 0
    delta_t = cfl * delta_x / abs(a)
    c = delta_t / delta_x
    sol = np.array(ics)
    for _ in tqdm(np.arange(0, t_end, delta_t)):
        differences = np.roll(sol, -1) - sol
        denom = np.where(differences != 0., 1. / differences, 0.)
        if a > 0:
            numer = np.roll(differences, 1)
        else:
            numer = np.roll(differences, -1)
        flow_parameter = np.where((numer == 0.) & (denom == 0.), 1., numer * denom)
        phi = 1 - (1 - abs(cfl)) * limiter(flow_parameter)
        flux_left = a * sol
        flux_right = np.roll(flux_left, -1)
        waf_flux = 0.5 * (flux_left + flux_right) - 0.5 * np.sign(cfl) * phi * (flux_right - flux_left)
        sol -= c * waf_flux
        sol += c * np.roll(waf_flux, 1)
        t += delta_t
    return sol


if __name__ == "__main__":
    boxsize = 1
    numpart = 100
    delta_x = boxsize / numpart
    a = 1.
    t_end = 10.

    x = np.linspace(0., boxsize, numpart, endpoint=False) + 0.5 / numpart
    ics = [transform(box(numpart)), transform(triangle(numpart)), transform(wave(numpart))]

    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(12, 4))
    for ic, ax in zip(ics, axes):
        ax.plot(x, ic, ls="--", c="k", label="ICs")
        ax.plot(x, godunov(ic, t_end, a=a, delta_x=delta_x), label="Godunov")
        ax.plot(x, muscl_hancock(ic, t_end, a=a, delta_x=delta_x, limiter=slope_limiter_direct),
                label="MHM")
        ax.plot(x, waf(ic, t_end, a=a, delta_x=delta_x, limiter=flux_limiter_superbee), label="WAF")
    axes[-1].legend()
    plt.tight_layout()
    plt.show()
