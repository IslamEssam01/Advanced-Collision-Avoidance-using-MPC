import carla
import numpy as np
from scipy.optimize import minimize

class EnhancedTrajectoryTracker:
    def __init__(self, world, vehicle, look_ahead_distance=5.0, k_adaptive=0.01):
        """
        Enhanced trajectory tracking controller
        
        :param world: CARLA world instance
        :param vehicle: Ego vehicle
        :param look_ahead_distance: Look-ahead distance for path following
        :param k_adaptive: Adaptive gain for steering
        """
        
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        
        # Trajectory tracking parameters
        self.look_ahead_distance = look_ahead_distance
        self.k_adaptive = k_adaptive
        
        # Trajectory storage
        self.current_trajectory = None
        self.trajectory_index = 0
    
    def find_closest_point(self, current_location):
        """
        Find the closest point on the trajectory
        
        :param current_location: Current vehicle location
        :return: Closest trajectory point and its index
        """
        if self.current_trajectory is None or len(self.current_trajectory) == 0:
            return None, -1
        
        # Convert current location to numpy array
        current_pos = np.array([current_location.x, current_location.y])
        
        # Calculate distances to all trajectory points
        distances = np.linalg.norm(
            self.current_trajectory - current_pos, 
            axis=1
        )
        
        # Find index of closest point
        closest_index = np.argmin(distances)
        
        return self.current_trajectory[closest_index], closest_index
    
    def find_look_ahead_point(self, current_location):
        """
        Find the look-ahead point on the trajectory
        
        :param current_location: Current vehicle location
        :return: Look-ahead point
        """
        if self.current_trajectory is None:
            return None
        
        # Convert current location to numpy array
        current_pos = np.array([current_location.x, current_location.y])
        
        # Start search from the closest point
        _, start_index = self.find_closest_point(current_location)
        
        # Search for look-ahead point
        for i in range(start_index, len(self.current_trajectory)):
            look_ahead_point = self.current_trajectory[i]
            distance = np.linalg.norm(look_ahead_point - current_pos)
            
            if distance >= self.look_ahead_distance:
                return look_ahead_point
        
        # If no point found, return the last trajectory point
        return self.current_trajectory[-1]
    
    def calculate_steering_angle(self, current_location, look_ahead_point):
        """
        Calculate steering angle using Pure Pursuit method with adaptive gains
        
        :param current_location: Current vehicle location
        :param look_ahead_point: Look-ahead point on trajectory
        :return: Steering angle
        """
        # Get vehicle transform
        vehicle_transform = self.vehicle.get_transform()
        
        # Convert points to numpy arrays
        vehicle_pos = np.array([current_location.x, current_location.y])
        target_pos = look_ahead_point
        
        # Calculate heading
        vehicle_heading = vehicle_transform.rotation.yaw
        
        # Calculate target angle
        angle_to_target = np.arctan2(
            target_pos[1] - vehicle_pos[1], 
            target_pos[0] - vehicle_pos[0]
        )
        angle_to_target_deg = np.degrees(angle_to_target)
        
        # Calculate angle difference
        angle_diff = (angle_to_target_deg - vehicle_heading + 180) % 360 - 180
        
        # Adaptive steering gain
        steering_gain = self.k_adaptive * angle_diff
        
        # Limit steering
        return max(min(steering_gain, 1.0), -1.0)
    
    def track_trajectory(self, trajectory):
        """
        Track the given trajectory with collision risk mitigation

        :param trajectory: Numpy array of trajectory points
        """
        # Store trajectory
        self.current_trajectory = trajectory
        self.trajectory_index = 0

        # Get current vehicle location
        current_location = self.vehicle.get_location()

        # Find look-ahead point
        look_ahead_point = self.find_look_ahead_point(current_location)

        if look_ahead_point is None:
            return

        # Calculate control
        control = carla.VehicleControl()

        # Steering
        control.steer = self.calculate_steering_angle(current_location, look_ahead_point)

        # Speed control
        current_speed = self.vehicle.get_velocity().length()
        max_speed = 30.0  # Maximum desired speed

        # Collision risk assessment
        collision_risk_factor = self._assess_collision_risk()

        # Adaptive throttle based on curvature and distance
        distance_to_target = np.linalg.norm(
            look_ahead_point - np.array([current_location.x, current_location.y])
        )

        # Base throttle
        base_throttle = 0.35

        # Speed factor
        speed_factor = 1.0 - (current_speed / max_speed)

        # Curvature adjustment
        curvature_factor = 1.0 - (0.8 * abs(control.steer))

        # Dynamic throttle with collision risk reduction
        dynamic_throttle = (
            base_throttle * 
            (0.4 * speed_factor + 
             0.3 * (distance_to_target / 200.0) + 
             0.3 * curvature_factor)
        )

        # Reduce throttle based on collision risk
        reduced_throttle = dynamic_throttle * (1 - collision_risk_factor)

        # Brake and throttle control
        if current_speed > max_speed:
            control.brake = min((current_speed - max_speed) / 10.0, 1.0)
            control.throttle = 0.0
        else:
            control.throttle = max(min(reduced_throttle, 1.0), 0.1)
            control.brake = 0.0  # Light braking if high risk

#         print(collision_risk_factor,control.throttle,control.brake)
        # Apply control
        self.vehicle.apply_control(control)

    def _assess_collision_risk(self):
        """
        Assess collision risk by checking nearby vehicles

        :return: Collision risk factor between 0 and 1
        """
        nearby_vehicles = self.world.get_actors().filter('vehicle*')
        current_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        forward_vector = vehicle_transform.get_forward_vector()
        
        collision_risks = []

        for vehicle in nearby_vehicles:
            if vehicle.id == self.vehicle.id:
                continue

            vehicle_location = vehicle.get_location()
            distance = current_location.distance(vehicle_location)

            # Progressive risk calculation
            base_risk = 1.0

            # Consider vehicle velocity and direction
            vehicle_velocity = vehicle.get_velocity()
            velocity_magnitude = vehicle_velocity.length()

            # Directional risk calculation with smoother transition
            point_to_vehicle = np.array([
                vehicle_location.x - current_location.x,
                vehicle_location.y - current_location.y
            ])

            # Normalize vectors
            forward_vec = np.array([forward_vector.x, forward_vector.y])
            forward_vec /= np.linalg.norm(forward_vec)
            point_to_vehicle /= np.linalg.norm(point_to_vehicle)

            # Smooth angle calculation
            angle_cos = np.dot(forward_vec, point_to_vehicle)

            if angle_cos > 0.5:
                # Define risk zones
                risk_zones = [
                    (2.0, 1.0),   # Very close (high risk)
                    (5.0, 0.5),   # Close (moderate risk)
                    (10.0, 0.3),  # Medium distance (low risk)
                    (15.0, 0.1),  # Far distance (low risk)

                ]

                for threshold, risk_factor in risk_zones:
                    if distance < threshold:
                        collision_risks.append(risk_factor)
                        break

        # Return maximum risk or 0 if no risks
        return max(collision_risks) if collision_risks else 0.0