import carla
import numpy as np
import random
from TrajectoryTracker import EnhancedTrajectoryTracker
from scipy.optimize import minimize

class RealTimeRoadMPC:
    def __init__(self, world, vehicle, target_location, horizon=12):
        """
        Initialize Real-Time Road-Constrained MPC Controller
        
        :param world: CARLA world instance
        :param vehicle: Ego vehicle
        :param target_location: Final destination location
        :param horizon: Prediction horizon steps
        """
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        self.target_location = target_location
        self.horizon = horizon
        self.trajectory_tracker = EnhancedTrajectoryTracker(world, vehicle)
        self.base_horizon = horizon
        self.previous_trajectory = None
        
        # Get current road waypoint
        self.current_waypoint = self.map.get_waypoint(
            vehicle.get_location(), 
            project_to_road=True
        )
        
    
    def calculate_adaptive_horizon(self):
        current_speed = self.vehicle.get_velocity().length()
        road_curvature = self._estimate_road_curvature()

        # Speed-based horizon
        speed_factor = current_speed / 50.0

        # Curvature-based horizon adjustment
        curvature_factor = road_curvature * 2.0

        adaptive_horizon = int(self.base_horizon * (1 + speed_factor + curvature_factor))
        return min(max(adaptive_horizon, 5), 20)  # Clamp between 5 and 20

    def _estimate_road_curvature(self):
        """
        Estimate road curvature by analyzing waypoint angles
        """
        current_wp = self.map.get_waypoint(self.vehicle.get_location())
        look_ahead_wps = current_wp.next(20)  # Look 20 meters ahead

        if len(look_ahead_wps) < 2:
            return 0

        # Calculate angle changes between waypoints
        angle_changes = [
            abs(wp1.transform.rotation.yaw - wp2.transform.rotation.yaw)
            for wp1, wp2 in zip(look_ahead_wps[:-1], look_ahead_wps[1:])
        ]

        return np.mean(angle_changes) / 180.0  # Normalize to [0, 1]
    
    def get_road_waypoints(self, start_waypoint, look_ahead=300):
        """
        Get future waypoints that progressively move towards the target location
        Strictly prioritize driving lanes
        """
        # Convert target location to waypoint
        target_wp = self.map.get_waypoint(
            carla.Location(
                x=self.target_location.x, 
                y=self.target_location.y, 
                z=start_waypoint.transform.location.z
            )
        )

        waypoints = [start_waypoint]
        current_wp = start_waypoint

        for _ in range(look_ahead):
            # Get next possible waypoints
            next_wps = current_wp.next(0.5)  # 1-meter steps
            if not next_wps:
                break

            # Strictly filter for driving lanes
            driving_wps = [
                wp for wp in next_wps 
                if wp.lane_type == carla.LaneType.Driving
            ]

            # If no driving lanes, use the original method
            if not driving_wps:
                break

            # Select waypoint that minimizes distance to target
            best_next_wp = min(driving_wps, key=lambda wp: (
                wp.transform.location.distance(target_wp.transform.location)
            ))

            # Break if we're getting further from the target
            if best_next_wp.transform.location.distance(target_wp.transform.location) > \
               current_wp.transform.location.distance(target_wp.transform.location):
                break

            current_wp = best_next_wp
            waypoints.append(current_wp)

            # Early stopping if close to target
            if current_wp.transform.location.distance(target_wp.transform.location) < 10:
                break

        return waypoints
    
    def _waypoint_to_vector(self, waypoint):
        """
        Convert waypoint to 2D vector
        
        :param waypoint: CARLA waypoint
        :return: 2D numpy array of waypoint location
        """
        loc = waypoint.transform.location
        return np.array([loc.x, loc.y])
    
    def smooth_trajectory(self, trajectory, smoothing_factor=0.1):
        """
        Enhanced trajectory smoothing with more conservative approach
        """
        smoothed = np.copy(trajectory)
        for i in range(1, len(trajectory) - 1):
            # Less aggressive smoothing
            smoothed[i] = (
                trajectory[i-1] * smoothing_factor + 
                trajectory[i] * (1 - 2 * smoothing_factor) + 
                trajectory[i+1] * smoothing_factor
            )
        return smoothed
    
    
    def calculate_trajectory_cost(self, candidate_trajectory, road_waypoints,previous_trajectory):
        """
        Comprehensive trajectory cost calculation

        :param candidate_trajectory: Proposed trajectory waypoints
        :param road_waypoints: Reference road waypoints
        :param previous_trajectory: Previous trajectory for continuity
        :return: Total cost of the trajectory
        """
        # Reshape trajectory if needed
        if candidate_trajectory.ndim == 1:
            candidate_trajectory = candidate_trajectory.reshape(-1, 2)

        # Convert target location to numpy array
        target = np.array([self.target_location.x, self.target_location.y])

        # Initial vehicle location
        start_point = np.array([
            self.vehicle.get_location().x, 
            self.vehicle.get_location().y
        ])

        # Cost components with more detailed calculation
        components = {
            'target_proximity': 0.0,
            'road_deviation': 0.0,
            'smoothness': 0.0,
            'collision_risk': 0.0,
            'trajectory_continuity': 0.0,
        }

        # Adaptive weights based on scenario
        weights = {
            'target_proximity': 600,
            'road_deviation': 500,  # Increased weight
            'smoothness': 200,
            'collision_risk': 500000,  # Increased collision risk weight
            'trajectory_continuity': 30,
        }
        
        # Target Proximity Cost (Enhanced)
        def calculate_advanced_target_proximity(trajectory):
            # Final point distance to target
            final_point_distance = np.linalg.norm(trajectory[-1] - target)

            # Progressive proximity calculation
            cumulative_proximity_bonus = 0
            for i, point in enumerate(trajectory):
                # Distance to target for each point
                point_distance = np.linalg.norm(point - target)

                # Bonus for getting closer to target
                if i > 0:
                    prev_point_dist = np.linalg.norm(trajectory[i-1] - target)
                    if point_distance < prev_point_dist:
                        # Reward for progressive approach
                        cumulative_proximity_bonus += (prev_point_dist - point_distance) * 5
                    else:
                        # Penalty for moving away
                        cumulative_proximity_bonus -= (point_distance - prev_point_dist) * 20

            return final_point_distance - cumulative_proximity_bonus

        # Advanced Road Deviation Cost
        def advanced_road_deviation_cost(trajectory):
            """
            Advanced road deviation cost calculation with multiple sophistication layers

            :param trajectory: Numpy array of trajectory points
            :return: Comprehensive road deviation cost
            """
            total_deviation = 0
            lane_penalties = {
                carla.LaneType.Driving: 1.0,     # Standard driving lane
                carla.LaneType.Sidewalk: 10000.0,   # High penalty for sidewalks
                carla.LaneType.Shoulder: 5.0,    # Moderate penalty for shoulders
                carla.LaneType.Parking: 800.0,     # High penalty for parking lanes
                carla.LaneType.Bidirectional: 3.0,  # Moderate penalty for bidirectional lanes
                carla.LaneType.Biking : 1000.0
            }

            # Adaptive lane width calculation
            def calculate_lane_boundaries(waypoint):
                """
                Calculate precise lane boundaries considering lane type and width
                """
                # Base lane width
                base_width = waypoint.lane_width or 3.5

                # Adjust width based on lane type
                if waypoint.lane_type == carla.LaneType.Driving:
                    lane_width = base_width
                elif waypoint.lane_type == carla.LaneType.Shoulder:
                    lane_width = base_width * 0.5
                else:
                    lane_width = base_width * 0.75

                return lane_width

            # Enhanced deviation calculation
            for i, point in enumerate(trajectory):
                if i >= len(road_waypoints):
                    break

                road_wp = road_waypoints[i]

                # Lane center calculation
                lane_center = np.array([
                    road_wp.transform.location.x,
                    road_wp.transform.location.y
                ])

                # Calculate lane boundaries
                lane_width = calculate_lane_boundaries(road_wp)

                # Calculate lateral deviation
                lateral_deviation = np.linalg.norm(point - lane_center)

                # Exponential deviation cost with sophisticated penalties
                deviation_cost = 0

                # Lane type specific penalty
                lane_type_penalty = lane_penalties.get(
                    road_wp.lane_type, 
                    2.0  # Default penalty for unknown lane types
                )

                # Progressive deviation cost
                if lateral_deviation <= lane_width / 2:
                    # Within lane: minimal cost
                    deviation_cost = lateral_deviation ** 2 * 0.1
                else:
                    # Outside lane: exponential cost
                    outside_distance = lateral_deviation - (lane_width / 2)
                    deviation_cost = (
                        (outside_distance ** 3) *  # Cubic growth for significant deviations
                        lane_type_penalty *        # Lane type specific multiplier
                        10000                      # Base multiplier for lane deviation
                    )

                # Consider road curvature
                if i > 0 and i < len(road_waypoints) - 1:
                    # Analyze angle between consecutive waypoints
                    prev_wp = road_waypoints[i-1]
                    next_wp = road_waypoints[i+1]

                    # Calculate curvature penalty
                    angle_change = abs(
                        prev_wp.transform.rotation.yaw - 
                        next_wp.transform.rotation.yaw
                    )
                    curvature_factor = 1 + (angle_change / 180.0)  # Normalize angle change

                    # Apply curvature factor to deviation cost
                    deviation_cost *= curvature_factor

                # Accumulate total deviation cost
                total_deviation += deviation_cost

            return total_deviation

        def advanced_collision_risk(trajectory):
            total_collision_risk = 0
            nearby_vehicles = self.world.get_actors().filter('vehicle*')
            
    
            # Get current vehicle's heading
            vehicle_transform = self.vehicle.get_transform()
            forward_vector = vehicle_transform.get_forward_vector()

            for i, point in enumerate(trajectory):
                location = carla.Location(x=point[0], y=point[1], z=0.5)

                for vehicle in nearby_vehicles:
                    if vehicle.id == self.vehicle.id:
                        continue

                    vehicle_location = vehicle.get_location()
                    distance = location.distance(vehicle_location)

                    # Progressive risk calculation
                    base_risk = 1.0

                    # Consider vehicle velocity and direction
                    vehicle_velocity = vehicle.get_velocity()
                    velocity_magnitude = vehicle_velocity.length()

                    # Directional risk calculation with smoother transition
                    point_to_vehicle = np.array([
                        vehicle_location.x - point[0],
                        vehicle_location.y - point[1]
                    ])

                    # Normalize vectors
                    forward_vec = np.array([forward_vector.x, forward_vector.y])
                    forward_vec /= np.linalg.norm(forward_vec)
                    point_to_vehicle /= np.linalg.norm(point_to_vehicle)

                    # Smooth angle calculation
                    angle_cos = np.dot(forward_vec, point_to_vehicle)

                    # Adaptive risk zones with gradual transitions
                    if angle_cos > 0.5:  # Front zone
                        front_zones = [
                            (8.0, 50000, 2.0),   # Very Close Zone
                            (10.0, 10000, 1.3),    # Close Zone
                            (12.0, 5000, 1.0),   # Medium Zone
                            (15.0, 2000, 0.7)    # Far Zone
                        ]

                        for threshold, multiplier,factor in front_zones:
                            if distance < threshold:
                                # Smoother exponential risk
                                total_collision_risk += (
                                    base_risk * 
                                    np.exp((threshold - distance) / threshold) *
#                                     np.exp((threshold - distance) / 2) * 
                                    multiplier * factor *
                                    (1 + velocity_magnitude * 0.2)  # Consider velocity
                                )
                                break
                    else:  # Side/rear zones
                        side_zones = [
                            (1.5, 10000),    # Close zone
                            (3.5, 5000)      # Medium zone
                        ]

                        for threshold, multiplier in side_zones:
                            if distance < threshold:
                                total_collision_risk += (
                                    base_risk * 
                                    np.exp((threshold - distance) / threshold) * 
#                                     np.exp((threshold - distance) / 2) * 
                                    multiplier  * 
                                    (1 + velocity_magnitude * 0.1)  # Consider velocity
                                )
                                break
                                
            return total_collision_risk

        # Trajectory Continuity Cost
        def trajectory_continuity_cost(trajectory, previous_trajectory):
            if previous_trajectory is None or len(previous_trajectory) == 0:
                return 0

            # Pad or truncate the previous trajectory to match the current trajectory length
            if len(previous_trajectory) > len(trajectory):
                prev_traj = previous_trajectory[:len(trajectory)]
            else:
                prev_traj = np.pad(
                    previous_trajectory, 
                    ((0, len(trajectory) - len(previous_trajectory)), (0, 0)), 
                    mode='edge'
                )

            # Calculate deviation from previous trajectory
            continuity_cost = np.sum((trajectory - prev_traj) ** 2)

            return continuity_cost

        # Smoothness Cost
        def smoothness_cost(trajectory):
            smoothness = 0
            for i in range(1, len(trajectory)):
                # Penalize large changes between consecutive points
                segment_length = np.linalg.norm(trajectory[i] - trajectory[i-1])
                smoothness += segment_length ** 2
            return smoothness


        # Calculate individual cost components
        components['target_proximity'] = calculate_advanced_target_proximity(candidate_trajectory)
        components['road_deviation'] = advanced_road_deviation_cost(candidate_trajectory)
        components['smoothness'] = smoothness_cost(candidate_trajectory)
        components['collision_risk'] = advanced_collision_risk(candidate_trajectory)
        components['trajectory_continuity'] = trajectory_continuity_cost(
            candidate_trajectory, 
            previous_trajectory
        )

        # Calculate total weighted cost
        total_cost = sum(
            component * weights[name] 
            for name, component in components.items()
        )

        return total_cost
    
    def create_initial_trajectory(self, road_waypoints):
        """
        Create an initial trajectory guess based on road waypoints and previous trajectory

        :param road_waypoints: List of road waypoints to guide trajectory generation
        :return: Initial trajectory guess for optimization
        """
        current_location = self.vehicle.get_location()

        # Initialize initial guess
        initial_guess = np.zeros(self.horizon * 2)
        

        # If previous trajectory exists, use it as a starting point
        if self.previous_trajectory is not None and len(self.previous_trajectory) > 0:
            # Shift previous trajectory forward
            shifted_trajectory = np.roll(self.previous_trajectory, -2, axis=0)

            # Fill initial guess with shifted trajectory
            initial_guess[:min(len(shifted_trajectory.flatten()), len(initial_guess))] = \
                shifted_trajectory.flatten()[:min(len(shifted_trajectory.flatten()), len(initial_guess))]

        # Interpolate between current location and road waypoints
        for i in range(self.horizon):
            # Calculate interpolation factor
            t = (i + 1) / self.horizon

            # If road waypoints are available, use them for guidance
            if i < len(road_waypoints):
                road_loc = road_waypoints[i].transform.location

                # Weighted interpolation between current location and road waypoint
                x = (1-t) * current_location.x + t * road_loc.x
                y = (1-t) * current_location.y + t * road_loc.y

                # Update initial guess if not already set
                if initial_guess[i*2] == 0 and initial_guess[i*2 + 1] == 0:
                    initial_guess[i*2] = x
                    initial_guess[i*2 + 1] = y

            # Add some random noise to prevent getting stuck
            noise_scale = 0.5  # Adjust as needed
            if initial_guess[i*2] == 0 and initial_guess[i*2 + 1] == 0:
                initial_guess[i*2] = current_location.x + np.random.normal(0, noise_scale)
                initial_guess[i*2 + 1] = current_location.y + np.random.normal(0, noise_scale)

        return initial_guess
    
    def generate_optimal_trajectory(self):
        """
        Generate optimal trajectory using MPC optimization
        
        :return: Optimized trajectory waypoints
        """
        # Calculate adaptive horizon
        self.horizon = self.calculate_adaptive_horizon()
        
        # Get current location and road waypoints
        current_location = self.vehicle.get_location()
        current_waypoint = self.map.get_waypoint(current_location, project_to_road=True)
        road_waypoints = self.get_road_waypoints(current_waypoint)
        
        # Warm start with previous trajectory
        initial_guess = self.create_initial_trajectory(road_waypoints)
    
        
        for i in range(self.horizon):
            # Interpolate between current location and road waypoints
            if i < len(road_waypoints):
                road_loc = road_waypoints[i].transform.location
                t = (i + 1) / self.horizon
                initial_guess[i*2] = (1-t) * current_location.x + t * road_loc.x
                initial_guess[i*2 + 1] = (1-t) * current_location.y + t * road_loc.y
            else:
                # Use last known point if we run out of road waypoints
                initial_guess[i*2] = initial_guess[(i-1)*2]
                initial_guess[i*2 + 1] = initial_guess[(i-1)*2 + 1]
        
        initial_guess_smoothed = self.smooth_trajectory(initial_guess.reshape(-1, 2)).flatten()
    
        # Optimize with previous trajectory consideration
        result = minimize(
            lambda x: self.calculate_trajectory_cost(
                x, 
                road_waypoints, 
                previous_trajectory=self.previous_trajectory
            ), 
            initial_guess_smoothed, 
            method='Nelder-Mead',
            options={
                'maxiter': 100,
                'xatol': 1e-3,
#                 'disp': True
            }
        )
        
        # Apply final smoothing to the result
        trajectory = result.x.reshape(-1, 2)
        smoothed_trajectory = self.smooth_trajectory(trajectory)

        # Store current trajectory for next iteration
        self.previous_trajectory = smoothed_trajectory
        
        return self.previous_trajectory
    
    def apply_control(self, trajectory):
        """
        Apply vehicle control based on optimized trajectory
        
        :param trajectory: Optimized trajectory waypoints
        """
        self.trajectory_tracker.track_trajectory(trajectory)
        
    def visualize_trajectory_debug(self, trajectory):
        """
        Visualize trajectory using CARLA's built-in debug drawing

        :param trajectory: Numpy array of trajectory points
        """
        # Draw trajectory points
        for i in range(len(trajectory) - 1):
            start_point = carla.Location(x=trajectory[i][0], y=trajectory[i][1], z=0.5)
            end_point = carla.Location(x=trajectory[i+1][0], y=trajectory[i+1][1], z=0.5)

            # Draw lines between points
            self.world.debug.draw_line(
                start_point, 
                end_point, 
                thickness=0.1,  # Line thickness
                color=carla.Color(255, 0, 0),  # Red color
                life_time=0.1   # How long the line remains visible
            )

            # Optionally draw small spheres at each point
            self.world.debug.draw_point(
                start_point, 
                size=0.1, 
                color=carla.Color(0, 255, 0),  # Green points
                life_time=0.1
            )