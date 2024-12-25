import cv2
import numpy as np
import os
import carla

def setup_camera(world, vehicle,x=-10,z=3,pitch=-10):
    """
    Set up a camera sensor attached to the vehicle
    
    Args:
        world (carla.World): CARLA world instance
        vehicle (carla.Vehicle): Vehicle to attach camera to
    
    Returns:
        carla.Sensor: Camera sensor
    """
    blueprint_library = world.get_blueprint_library()
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    
    # Adjust camera parameters
    camera_bp.set_attribute('image_size_x', '1280')
    camera_bp.set_attribute('image_size_y', '720')
    camera_bp.set_attribute('fov', '90')
    
    # Camera location relative to vehicle (slightly elevated, centered)
    camera_transform = carla.Transform(
        carla.Location(x=x, z=z),  # Slightly to the right and above vehicle
        carla.Rotation(pitch=pitch)  # Slight downward angle
    )
    
    # Spawn the camera
    camera = world.spawn_actor(
        camera_bp, 
        camera_transform, 
        attach_to=vehicle
    )
    
    return camera

def create_video_writer(output_path, fps=30, width=1280, height=720):
    """
    Create a video writer for saving camera footage
    
    Args:
        output_path (str): Path to save the video
        fps (int): Frames per second
        width (int): Video width
        height (int): Video height
    
    Returns:
        cv2.VideoWriter: Video writing object
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, (width, height))

def save_camera_footage(world, vehicle,x=-10,z=3,pitch=-10, output_path='./vehicle_view.mp4'):
    """
    Set up a camera and record video of the simulation
    
    Args:
        world (carla.World): CARLA world instance
        vehicle (carla.Vehicle): Vehicle to attach camera to
        output_path (str): Path to save the video
    
    Returns:
        list: Captured image frames
    """
    # Setup camera
    camera = setup_camera(world, vehicle,x,z,pitch)
    
    # Create video writer
    video_writer = create_video_writer(output_path)
    
    # List to store frames (optional, if you want to process frames later)
    captured_frames = []
    
    # Callback for image processing
    def process_image(image):
        # Convert to numpy array
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        
        # Convert RGBA to BGR (for OpenCV)
        img = img[:, :, :3]
        
        # Write to video
        video_writer.write(img)
        
        # Optionally store frames
        captured_frames.append(img)
    
    # Listen to camera sensor
    camera.listen(process_image)
    
    return camera, video_writer, captured_frames
