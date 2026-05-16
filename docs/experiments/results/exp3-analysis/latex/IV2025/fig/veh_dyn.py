import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from matplotlib.patches import Wedge



def draw_arrow(ax, start, end, label=None, color="blue"):
    ax.annotate("",
                xy=end, xytext=start,
                arrowprops=dict(facecolor=color, shrink=0.05))
    if label:
        ax.text((start[0]+end[0])/2, (start[1]+end[1])/2, label,
                color=color, fontsize=10)

def main():
    # LaTeXモードを有効にする
    plt.rc('text', usetex=False)
    #plt.rc('font', family='serif')
    #plt.rc('text.latex.preamble', r'\usepackage{mathptmx}')

    fig, ax = plt.subplots(figsize=(8, 6))

    # Transformation for rotation
    rotation_angle = 35  # Rotate the car by 30 degrees
    trans = Affine2D().rotate_deg(rotation_angle) + ax.transData

    # Draw the car rectangle
    car_length = 3.5
    car_width = 2
    car_rect = plt.Rectangle((-car_length/2, -car_width/2), car_length, car_width,
                              edgecolor='black', facecolor='none', lw=1.5, transform=trans)
    ax.add_patch(car_rect)

    # Add center of gravity mark as a target symbol
    radius = 0.2
    ax.add_patch(Wedge((0, 0), radius, 0, 90, facecolor='black', transform=trans))
    ax.add_patch(Wedge((0, 0), radius, 90, 180, facecolor='white', edgecolor='black', transform=trans))
    ax.add_patch(Wedge((0, 0), radius, 180, 270, facecolor='black', transform=trans))
    ax.add_patch(Wedge((0, 0), radius, 270, 360, facecolor='white', edgecolor='black', transform=trans))

    # Draw the road with rotation
    road_rotation = Affine2D().rotate_deg(30) + ax.transData
    road_x = np.linspace(-6, 8, 500)
    road_y = 0.03 * road_x**2 - 0.8  # Slightly curved road
    ax.plot(road_x, road_y, color='gray', lw=1, zorder=-1, transform=road_rotation)
    ax.fill_between(road_x, road_y - 2.3, road_y + 2.3, color='lightgray', alpha=0.5, zorder=-2, transform=road_rotation)

    # Local axes x, y
    ax.arrow(0, 0, 3, 0, head_width=0.2, head_length=0.4, fc='k', ec='k', ls='-', transform=trans, lw = 0.5)
    ax.arrow(0, 0, 0, 2, head_width=0.2, head_length=0.4, fc='k', ec='k', ls='-', transform=trans, lw = 0.5)
    ax.text(3, -0.5, r"x", fontsize=20, color='k', transform=trans, fontname='Times New Roman', style='italic')
    ax.text(0.4, 2, r'y', fontsize=20, color='k', transform=trans, fontname='Times New Roman', style='italic')

    # Tires (simplified rectangles)
    tire_width = 0.5
    tire_length = 1.0
    ax.add_patch(plt.Rectangle((-2.3, -1.2), tire_length, tire_width, angle=0, color='gray', transform=trans))
    ax.add_patch(plt.Rectangle((-2.3, 0.8), tire_length, tire_width, angle=0, color='gray', transform=trans))
    ax.add_patch(plt.Rectangle((1.2, -1.2), tire_length, tire_width, angle=-15, color='gray', transform=trans))
    ax.add_patch(plt.Rectangle((1.2, 0.8), tire_length, tire_width, angle=-15, color='gray', transform=trans))

    # steering wheel
    a = 0.13 # 長軸
    b = 0.3  # 短軸
    theta = np.linspace(0, 2 * np.pi, 100)
    x = a * np.cos(theta)
    y = b * np.sin(theta)
    rotation_angle = np.pi / 6  # 30度
    x_rotated = x * np.cos(rotation_angle) - y * np.sin(rotation_angle)
    y_rotated = x * np.sin(rotation_angle) + y * np.cos(rotation_angle)
    plt.plot(x_rotated + 1.0, y_rotated + 0.2, color='k')
    # cross
    plt.plot([-a + 1.2, a + 0.8], [-a + .1, a + 0.3], color='k', linestyle='-')
    plt.plot([-a + 0.95, a + 1.05], [a + .28, -a + 0.12], color='k', linestyle='-')
    plt.plot([1.03, 1.79], [0.20, 0.50], color='k', linestyle='-')

    # arrow for steering wheel
    arc_angle_start = 0  # 開始角度（ラジアン）
    arc_angle_end = np.pi / 7  # 終了角度（ラジアン）
    
    arc_theta = np.linspace(arc_angle_start, arc_angle_end, 100)
    arc_radius = 0.5
    arc_x = arc_radius * np.cos(arc_theta)
    arc_y = arc_radius * np.sin(arc_theta)
    
    # 回転を適用
    rotation_angle = np.pi * 1.1 # 30度
    arc_x_rotated = arc_x * np.cos(rotation_angle) - arc_y * np.sin(rotation_angle)
    arc_y_rotated = arc_x * np.sin(rotation_angle) + arc_y * np.cos(rotation_angle)

    # 円弧のプロット
    plt.plot(arc_x_rotated + 1.2, arc_y_rotated + 0.35, color='k', linewidth=1, label='Arc')
    plt.arrow(
        arc_x_rotated[0] + 1.2, arc_y_rotated[0] + 0.35,  # 矢印の始点
        - (arc_x_rotated[1] - arc_x_rotated[0]) * 0.3,  # x方向の矢印の長さ
        - (arc_y_rotated[1] - arc_y_rotated[0]) * 0.3,  # y方向の矢印の長さ
        head_width=0.1, head_length=0.15, fc='k'
    )
    plt.arrow(
        arc_x_rotated[-1] + 1.2, arc_y_rotated[-1] + 0.35,  # 矢印の始点
        (arc_x_rotated[-1] - arc_x_rotated[-2]) * 0.3,  # x方向の矢印の長さ
        (arc_y_rotated[-1] - arc_y_rotated[-2]) * 0.3,  # y方向の矢印の長さ
        head_width=0.1, head_length=0.15, fc='k'
    )
    #ax.text(.3, -.4, r'$\theta$', fontsize=20, color='k', transform=trans, fontname='Times New Roman')
    ax.text(.55, -.18, 'θ', fontsize=20, color='k', fontname='Times New Roman')

    # 両端に矢印がある直線
    line_start = [0.24, -0.45]  # 直線の始点
    line_end = [0.16, -.3]      # 直線の終点
    #plt.plot(line_start, line_end, color='k', linewidth=1)
    plt.plot(
        [line_start[0], line_end[0]],
        [line_start[1], line_end[1]],
        color='k', linestyle='-', linewidth=1
    )
    plt.arrow(
        line_start[0], line_start[1],  # 矢印の始点
        -(line_end[0] - line_start[0]) * 0.5,  # x方向の長さ
        -(line_end[1] - line_start[1]) * 0.5,  # y方向の長さ
        head_width=0.1, head_length=0.2, fc='k', ec=None, label='Line with Arrows'
    )
    plt.arrow(
        line_end[0], line_end[1],  # 逆方向の矢印の始点
        (line_end[0] - line_start[0]) * 0.5,  # x方向の長さ
        (line_end[1] - line_start[1]) * 0.5,  # y方向の長さ
        head_width=0.1, head_length=0.2, fc='k', ec=None
    )
    #ax.text(-.5, -.5, r'$\delta$', fontsize=20, color='k', transform=trans, fontname='Times New Roman')
    ax.text(-.5, -.5, 'δ', fontsize=20, color='k', transform=trans, fontname='Times New Roman')

    # Styling
    ax.set_xlim(-3.0, 3.5)
    ax.set_ylim(-3.0, 3.0)
    ax.set_aspect('equal', adjustable='datalim')
    ax.axis('off')  # Turn off axis lines, ticks, and labels
    plt.savefig("vehicle_dynamics.pdf", format="pdf", bbox_inches="tight")

    plt.show()

if __name__ == "__main__":
    main()

