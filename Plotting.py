import os
import matplotlib.pyplot as plt

def save_individual_plots(time_data, speed_data, throttle_data, x_position_data, y_position_data, steering_data, output_dir='carla_plots'):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Speed plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, speed_data)
    plt.title('Vehicle Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_plot.png'))
    plt.close()

    # Throttle plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, throttle_data)
    plt.title('Throttle')
    plt.xlabel('Time (s)')
    plt.ylabel('Throttle Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'throttle_plot.png'))
    plt.close()

    # X,Y Position plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, x_position_data, label='X Position')
    plt.plot(time_data, y_position_data, label='Y Position')
    plt.title('X,Y Position over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('X,Y Position')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_plot.png'))
    plt.close()

    # Steering Angle plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_data, steering_data)
    plt.title('Steering Angle')
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'steering_plot.png'))
    plt.close()

    print(f"Individual plots have been saved in the '{output_dir}' directory.")
