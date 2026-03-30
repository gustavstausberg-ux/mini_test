import math
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider

Mode = Literal["range", "trajectory", "comparison"]

G = 9.81
CURRENT_MODE: Mode = "range"


def required_initial_speed(target_range: float, angles_deg: np.ndarray, g: float = G) -> np.ndarray:
    """Calculate the initial speed required to reach a target range for each launch angle."""
    angles_rad = np.radians(angles_deg)
    sin_two_theta = np.sin(2 * angles_rad)

    speeds = np.full_like(angles_deg, np.nan, dtype=float)
    valid = sin_two_theta > 0
    speeds[valid] = np.sqrt(target_range * g / sin_two_theta[valid])
    return speeds


def max_height(v0: np.ndarray, angles_rad: np.ndarray, g: float = G) -> np.ndarray:
    """Maximum height of the projectile."""
    return (v0**2 * np.sin(angles_rad) ** 2) / (2 * g)


def flight_time(v0: np.ndarray, angles_rad: np.ndarray, g: float = G) -> np.ndarray:
    """Total flight time until the projectile returns to the launch height."""
    return (2 * v0 * np.sin(angles_rad)) / g


def horizontal_range(v0: np.ndarray, angles_rad: np.ndarray, g: float = G) -> np.ndarray:
    """Horizontal range of the projectile."""
    return (v0**2 * np.sin(2 * angles_rad)) / g


def trajectory_points(v0: float, angle_deg: float, num_points: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """Return x and y coordinates for a projectile trajectory."""
    angle_rad = math.radians(angle_deg)
    total_time = flight_time(np.array([v0]), np.array([angle_rad]))[0]
    times = np.linspace(0, total_time, num_points)

    x = v0 * math.cos(angle_rad) * times
    y = v0 * math.sin(angle_rad) * times - 0.5 * G * times**2
    return x, y


def choose_start_mode() -> Mode:
    """Ask the user which mode should be shown first."""
    print("Projectile Motion Explorer")
    print("Choose a mode to open first:")
    print("1 - Range analysis")
    print("2 - Trajectory plot")
    print("3 - Angle comparison")

    options: dict[str, Mode] = {
        "1": "range",
        "2": "trajectory",
        "3": "comparison",
    }

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in options:
            return options[choice]
        print("Please enter 1, 2, or 3.\n")


def setup_figure() -> tuple[plt.Figure, plt.Axes]:
    """Create the main figure and plotting area."""
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.3)
    return fig, ax


def create_slider(fig: plt.Figure, position: list[float], label: str, vmin: float, vmax: float, valinit: float) -> Slider:
    """Create a slider on the figure."""
    slider_ax = fig.add_axes(position)
    return Slider(slider_ax, label, vmin, vmax, valinit=valinit)


def hide_sliders(sliders: list[Slider]) -> None:
    """Hide all slider axes."""
    for slider in sliders:
        slider.ax.set_visible(False)


def show_sliders(sliders: list[Slider]) -> None:
    """Show selected slider axes."""
    for slider in sliders:
        slider.ax.set_visible(True)


def draw_range_mode(ax: plt.Axes, target_range: float) -> None:
    """Draw the range-analysis mode."""
    ax.clear()

    angles_deg = np.linspace(1, 89, 400)
    required_speeds = required_initial_speed(target_range, angles_deg)
    optimal_angle_deg = 45.0
    optimal_speed = math.sqrt(target_range * G)

    ax.plot(angles_deg, required_speeds, linewidth=2)
    ax.scatter(optimal_angle_deg, optimal_speed, s=80, zorder=3)
    ax.axvline(optimal_angle_deg, linestyle="--", linewidth=1)
    ax.annotate(
        f"Optimal angle: {optimal_angle_deg:.0f}°\nRequired speed: {optimal_speed:.2f} m/s",
        xy=(optimal_angle_deg, optimal_speed),
        xytext=(53, optimal_speed + 2),
        arrowprops={"arrowstyle": "->"},
    )

    ax.set_title(f"Required Initial Speed for a Range of {target_range:.2f} m")
    ax.set_xlabel("Launch angle (degrees)")
    ax.set_ylabel("Required initial speed (m/s)")
    ax.grid(True, alpha=0.3)

    summary = (
        f"Target range: {target_range:.2f} m\n"
        f"Minimum possible speed: {optimal_speed:.2f} m/s\n"
        f"Achieved at: 45°"
    )
    ax.text(
        0.02,
        0.97,
        summary,
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "alpha": 0.15},
    )


def draw_trajectory_mode(ax: plt.Axes, speed: float, angle_deg: float) -> None:
    """Draw the trajectory mode."""
    ax.clear()

    x, y = trajectory_points(speed, angle_deg)
    angle_rad = math.radians(angle_deg)
    total_time = flight_time(np.array([speed]), np.array([angle_rad]))[0]
    distance = horizontal_range(np.array([speed]), np.array([angle_rad]))[0]
    peak = max_height(np.array([speed]), np.array([angle_rad]))[0]

    ax.plot(x, y, linewidth=2)
    ax.set_title("Projectile Trajectory")
    ax.set_xlabel("Horizontal distance (m)")
    ax.set_ylabel("Height (m)")
    ax.grid(True, alpha=0.3)
    max_speed = 50.0
    max_angle_rad = math.radians(89.0)
    max_range_limit = horizontal_range(np.array([max_speed]), np.array([math.radians(45.0)]))[0]
    max_height_limit = max_height(np.array([max_speed]), np.array([max_angle_rad]))[0]

    ax.set_xlim(0, max_range_limit * 1.05)
    ax.set_ylim(0, max_height_limit * 1.05)

    summary = (
        f"Initial speed: {speed:.2f} m/s\n"
        f"Launch angle: {angle_deg:.2f}°\n"
        f"Range: {distance:.2f} m\n"
        f"Max height: {peak:.2f} m\n"
        f"Flight time: {total_time:.2f} s"
    )
    ax.text(
        0.02,
        0.97,
        summary,
        transform=ax.transAxes,
        va="top",
        bbox={"boxstyle": "round", "alpha": 0.15},
    )


def draw_comparison_mode(ax: plt.Axes, speed: float, scale: float) -> None:
    """Draw the angle-comparison mode."""
    ax.clear()

    angles_deg = np.linspace(1, 89, 400)
    angles_rad = np.radians(angles_deg)
    speed_array = np.full_like(angles_rad, speed)

    heights = max_height(speed_array, angles_rad)
    times = flight_time(speed_array, angles_rad)
    ranges = horizontal_range(speed_array, angles_rad)

    ax.plot(angles_deg, heights, label="Maximum height", linewidth=2)
    ax.plot(angles_deg, times, label="Flight time", linewidth=2)
    ax.plot(angles_deg, ranges, label="Range", linewidth=2)

    best_range_index = np.argmax(ranges)
    best_range_angle = angles_deg[best_range_index]
    best_range_value = ranges[best_range_index]
    ax.scatter(best_range_angle, best_range_value, s=80, zorder=3)
    ax.annotate(
        f"Max range at {best_range_angle:.1f}°",
        xy=(best_range_angle, best_range_value),
        xytext=(best_range_angle + 5, best_range_value * 0.9),
        arrowprops={"arrowstyle": "->"},
    )

    ax.set_title(f"Angle Comparison at {speed:.2f} m/s")
    ax.set_xlabel("Launch angle (degrees)")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    max_speed = 50.0
    full_speed_array = np.full_like(angles_rad, max_speed)
    max_height_limit = np.max(max_height(full_speed_array, angles_rad))
    max_time_limit = np.max(flight_time(full_speed_array, angles_rad))
    max_range_limit = np.max(horizontal_range(full_speed_array, angles_rad))
    overall_limit = max(max_height_limit, max_time_limit, max_range_limit)

    ax.set_xlim(1, 89)
    ax.set_ylim(0, overall_limit / scale)


def main() -> None:
    global CURRENT_MODE

    CURRENT_MODE = choose_start_mode()

    fig, ax = setup_figure()

    target_range_slider = create_slider(fig, [0.15, 0.18, 0.7, 0.03], "Target range (m)", 1.0, 100.0, 20.0)
    speed_slider = create_slider(fig, [0.15, 0.18, 0.7, 0.03], "Initial speed (m/s)", 1.0, 50.0, 20.0)
    angle_slider = create_slider(fig, [0.15, 0.12, 0.7, 0.03], "Launch angle (deg)", 1.0, 89.0, 45.0)
    comparison_speed_slider = create_slider(fig, [0.15, 0.18, 0.7, 0.03], "Initial speed (m/s)", 1.0, 50.0, 20.0)
    comparison_scale_slider = create_slider(fig, [0.15, 0.12, 0.7, 0.03], "Y-zoom", 1.0, 50.0, 1.0)

    all_sliders = [target_range_slider, speed_slider, angle_slider, comparison_speed_slider, comparison_scale_slider]

    button_ax_range = fig.add_axes([0.15, 0.04, 0.18, 0.045])
    button_ax_trajectory = fig.add_axes([0.39, 0.04, 0.18, 0.045])
    button_ax_comparison = fig.add_axes([0.63, 0.04, 0.18, 0.045])

    button_range = Button(button_ax_range, "Range analysis")
    button_trajectory = Button(button_ax_trajectory, "Trajectory plot")
    button_comparison = Button(button_ax_comparison, "Angle comparison")

    def redraw() -> None:
        hide_sliders(all_sliders)

        if CURRENT_MODE == "range":
            show_sliders([target_range_slider])
            draw_range_mode(ax, target_range_slider.val)
        elif CURRENT_MODE == "trajectory":
            show_sliders([speed_slider, angle_slider])
            draw_trajectory_mode(ax, speed_slider.val, angle_slider.val)
        elif CURRENT_MODE == "comparison":
            show_sliders([comparison_speed_slider, comparison_scale_slider])
            draw_comparison_mode(ax, comparison_speed_slider.val, comparison_scale_slider.val)

        fig.canvas.draw_idle()

    def switch_to_range(_event: object) -> None:
        global CURRENT_MODE
        CURRENT_MODE = "range"
        redraw()

    def switch_to_trajectory(_event: object) -> None:
        global CURRENT_MODE
        CURRENT_MODE = "trajectory"
        redraw()

    def switch_to_comparison(_event: object) -> None:
        global CURRENT_MODE
        CURRENT_MODE = "comparison"
        redraw()

    target_range_slider.on_changed(lambda _value: redraw())
    speed_slider.on_changed(lambda _value: redraw())
    angle_slider.on_changed(lambda _value: redraw())
    comparison_speed_slider.on_changed(lambda _value: redraw())
    comparison_scale_slider.on_changed(lambda _value: redraw())

    button_range.on_clicked(switch_to_range)
    button_trajectory.on_clicked(switch_to_trajectory)
    button_comparison.on_clicked(switch_to_comparison)

    redraw()
    plt.show()


if __name__ == "__main__":
    main()