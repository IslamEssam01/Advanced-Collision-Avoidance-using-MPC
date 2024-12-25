import time
import carla
from MPC import RealTimeRoadMPC 
from Recording import save_camera_footage 
from Plotting import save_individual_plots 

def get_forward_target_on_road(world, vehicle, distance=100):
    vehicle_transform = vehicle.get_transform()
    forward_vector = vehicle_transform.get_forward_vector()
    current_location = vehicle_transform.location
    
    # Calculate potential target
    potential_target = carla.Location(
        x=current_location.x + forward_vector.x * distance,
        y=current_location.y + forward_vector.y * distance,
        z=current_location.z
    )
    
    # Get nearest waypoint on the road
    map = world.get_map()
    target_waypoint = map.get_waypoint(potential_target, project_to_road=True)
    
    return target_waypoint.transform.location

def main():
    # CARLA setup
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    
    world = client.get_world()
       
    
    # Spawn vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
    obstacle_bp = blueprint_library.filter('vehicle.mercedes.coupe_2020')[0]
    spawn_points = world.get_map().get_spawn_points()
    spawn_point = spawn_points[0]
    
    collisions = 0
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    # Attach collision sensor
    collision_sensor_bp = blueprint_library.find('sensor.other.collision')
    collision_sensor = world.spawn_actor(
        collision_sensor_bp, 
        carla.Transform(), 
        attach_to=vehicle
    )

    # Spectator
    spectator = world.get_spectator()


    vehicle_transform = vehicle.get_transform()

# First Scenario
#     target_location = spawn_points[-2].location
#     obstacle_location =carla.Transform(carla.Location(
#         x=-41.7,
#         y=-10.5,
#         z=vehicle_transform.location.z
#     ),carla.Rotation(pitch=0.002452, yaw=-89.170891, roll=-0.009186))

# Second Scenario
    target_location = get_forward_target_on_road(world, vehicle,100)
    obstacle_location =carla.Transform(carla.Location(
        x=19.8,
        y=24.5,
        z=vehicle_transform.location.z
    ),carla.Rotation(pitch=0.000000, yaw=0.159198, roll=0.000000))

    obstacle = world.spawn_actor(obstacle_bp,obstacle_location)
    
    collision_detected = [False]

    # Collision event callback
    def on_collision(event):
        collision_detected[0] = True
        print(f"Collision detected with {event.other_actor.type_id}")

    # Attach collision callback
    collision_sensor.listen(on_collision)

     # Data logging lists
    time_data = []
    speed_data = []
    throttle_data = []
    x_position_data = []
    y_position_data = []
    steering_data = []

    
    # MPC Controller
    mpc_controller = RealTimeRoadMPC(world, vehicle, target_location)
    
    start_time = time.time()
    camera, video_writer, captured_frames = save_camera_footage(world, vehicle)
    top_camera, top_video_writer, top_captured_frames = save_camera_footage(world, vehicle,0,20,-90,"./top.mp4")
    # Simulation loop
    try:
        while True:
            if collision_detected[0]:
                collisions+=1
                print("Collision occurred! Ending simulation.")
                break

                
                
            # Get current time
            current_time = time.time() - start_time
            # Generate optimal trajectory
            trajectory = mpc_controller.generate_optimal_trajectory()

#             mpc_controller.print_debug_info()
#             Apply control
            mpc_controller.apply_control(trajectory)
    # Get vehicle data
            vehicle_transform = vehicle.get_transform()
            vehicle_velocity = vehicle.get_velocity()
            vehicle_control = vehicle.get_control()

            # Log data
            time_data.append(current_time)
            speed_data.append(vehicle_velocity.length())
            throttle_data.append(vehicle_control.throttle)
            x_position_data.append(vehicle_transform.location.x)
            y_position_data.append(vehicle_transform.location.y)
            steering_data.append(vehicle_control.steer)
            
            mpc_controller.visualize_trajectory_debug(trajectory)

            spectator_transform = carla.Transform(
                vehicle_transform.location + carla.Location(z=20),
                carla.Rotation(pitch=-90)
            )
            spectator.set_transform(spectator_transform)

            # Check target reached
            current_location = vehicle.get_location()
            if current_location.distance(target_location) < 10:
                print("Reached target!")
                break
    finally:     
        save_individual_plots(time_data, speed_data, throttle_data, x_position_data, y_position_data, steering_data, output_dir='plots')
        camera.stop()
        camera.destroy()
        video_writer.release()
        top_camera.stop()
        top_camera.destroy()
        top_video_writer.release()
        
        
        collision_sensor.stop()
        collision_sensor.destroy()
        vehicle.destroy()
        obstacle.destroy()
        
if __name__ == '__main__':
    main()