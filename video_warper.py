import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys

class MeshWarper:
    def __init__(self, video_path, mesh_file=None, rows=10, cols=10):
        """
        Initialize the mesh warper
        
        Args:
            video_path: Path to video file or camera index (0 for webcam)
            mesh_file: Path to mesh warp file (optional)
            rows: Number of mesh rows
            cols: Number of mesh columns
        """
        self.rows = rows
        self.cols = cols
        
        # Open video source
        if isinstance(video_path, int):
            self.cap = cv2.VideoCapture(video_path)
        else:
            self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError("Could not open video source")
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Initialize mesh
        if mesh_file:
            self.mesh = self.load_mesh(mesh_file)
        else:
            self.mesh = self.create_identity_mesh()
        
        # Texture ID
        self.texture_id = None
        
        # Initialize pygame and OpenGL
        self.init_gl()
        
    def load_mesh(self, mesh_file):
        """Load mesh from file (format: x y u v intensity)"""
        mesh = []
        try:
            with open(mesh_file, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            x, y, u, v = map(float, parts[:4])
                            intensity = float(parts[4]) if len(parts) > 4 else 1.0
                            mesh.append({
                                'x': x, 'y': y,
                                'u': u, 'v': v,
                                'i': intensity
                            })
        except FileNotFoundError:
            print(f"Mesh file {mesh_file} not found, using identity mesh")
            return self.create_identity_mesh()
        
        return mesh
    
    def create_identity_mesh(self):
        """Create an identity mesh (no warping)"""
        mesh = []
        for r in range(self.rows):
            for c in range(self.cols):
                x = (c / (self.cols - 1)) * 2.0 - 1.0  # -1 to 1
                y = (r / (self.rows - 1)) * 2.0 - 1.0  # -1 to 1
                u = c / (self.cols - 1)  # 0 to 1
                v = 1.0 - (r / (self.rows - 1))  # 0 to 1 (flipped for OpenGL)
                
                mesh.append({
                    'x': x, 'y': y,
                    'u': u, 'v': v,
                    'i': 1.0
                })
        return mesh
    
    def init_gl(self):
        """Initialize OpenGL context"""
        pygame.init()
        display = (self.width, self.height)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
        pygame.display.set_caption("Real-time Mesh Warping")
        
        # Set up orthographic projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        
        # Enable texturing
        glEnable(GL_TEXTURE_2D)
        
        # Create texture
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
    def update_texture(self, frame):
        """Update OpenGL texture with new frame"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip vertically for OpenGL
        frame_rgb = cv2.flip(frame_rgb, 0)
        
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.width, self.height, 
                     0, GL_RGB, GL_UNSIGNED_BYTE, frame_rgb)
    
    def draw_mesh(self):
        """Draw the warped mesh"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        
        # Draw mesh as triangle strips
        for r in range(self.rows - 1):
            glBegin(GL_TRIANGLE_STRIP)
            for c in range(self.cols):
                # Bottom vertex
                idx1 = r * self.cols + c
                m1 = self.mesh[idx1]
                glColor3f(m1['i'], m1['i'], m1['i'])
                glTexCoord2f(m1['u'], m1['v'])
                glVertex2f(m1['x'], m1['y'])
                
                # Top vertex
                idx2 = (r + 1) * self.cols + c
                m2 = self.mesh[idx2]
                glColor3f(m2['i'], m2['i'], m2['i'])
                glTexCoord2f(m2['u'], m2['v'])
                glVertex2f(m2['x'], m2['y'])
            glEnd()
    
    def apply_barrel_distortion(self, strength=0.3):
        """Apply barrel distortion effect to mesh"""
        for i, node in enumerate(self.mesh):
            # Get normalized coordinates (-1 to 1)
            u_norm = node['u'] * 2.0 - 1.0
            v_norm = node['v'] * 2.0 - 1.0
            
            # Calculate distance from center
            r = np.sqrt(u_norm**2 + v_norm**2)
            
            # Apply barrel distortion formula
            factor = 1.0 + strength * r**2
            
            # Update mesh positions
            self.mesh[i]['x'] = u_norm * factor
            self.mesh[i]['y'] = v_norm * factor
    
    def apply_pincushion_distortion(self, strength=0.3):
        """Apply pincushion distortion effect to mesh"""
        self.apply_barrel_distortion(-strength)
    
    def reset_mesh(self):
        """Reset to identity mesh"""
        self.mesh = self.create_identity_mesh()
    
    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("\nControls:")
        print("  B - Apply barrel distortion")
        print("  P - Apply pincushion distortion")
        print("  R - Reset mesh")
        print("  Q/ESC - Quit")
        print("  SPACE - Pause/Resume\n")
        
        paused = False
        
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        running = False
                    elif event.key == K_b:
                        self.apply_barrel_distortion(0.3)
                        print("Applied barrel distortion")
                    elif event.key == K_p:
                        self.apply_pincushion_distortion(0.3)
                        print("Applied pincushion distortion")
                    elif event.key == K_r:
                        self.reset_mesh()
                        print("Reset mesh")
                    elif event.key == K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Resumed")
            
            if not paused:
                # Read frame
                ret, frame = self.cap.read()
                if not ret:
                    # Loop video
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                
                # Update texture
                self.update_texture(frame)
            
            # Draw mesh
            self.draw_mesh()
            
            # Swap buffers
            pygame.display.flip()
            
            # Control frame rate
            clock.tick(self.fps)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        pygame.quit()

def main():
    # Example usage
    if len(sys.argv) > 1:
        video_source = sys.argv[1]
        if video_source.isdigit():
            video_source = int(video_source)
    else:
        # Default to webcam
        video_source = 0
    
    mesh_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    try:
        warper = MeshWarper(video_source, mesh_file, rows=20, cols=20)
        warper.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
