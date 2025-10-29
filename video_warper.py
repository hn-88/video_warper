import cv2
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import threading
import os

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
        self.video_path = video_path
        self.mesh_file = mesh_file
        
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
        
        # Window properties
        self.window_width = 800
        self.window_height = 600
        self.fullscreen = False
        
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
                lines = f.readlines()
                
                # Try to determine rows and cols from file
                # First, skip any header lines
                data_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
                
                if data_lines:
                    # Parse all mesh points
                    points = []
                    for line in data_lines:
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            x, y, u, v = map(float, parts[:4])
                            intensity = float(parts[4]) if len(parts) > 4 else 1.0
                            points.append({
                                'x': x, 'y': y,
                                'u': u, 'v': v,
                                'i': intensity
                            })
                    
                    # Try to auto-detect mesh dimensions
                    if len(points) > 0:
                        # Count unique x values for cols, unique y values for rows
                        unique_u = len(set(p['u'] for p in points))
                        unique_v = len(set(p['v'] for p in points))
                        
                        if unique_u * unique_v == len(points):
                            self.cols = unique_u
                            self.rows = unique_v
                            print(f"Detected mesh dimensions: {self.rows}x{self.cols}")
                    
                    return points
                    
        except FileNotFoundError:
            print(f"Mesh file {mesh_file} not found, using identity mesh")
            return self.create_identity_mesh()
        except Exception as e:
            print(f"Error loading mesh: {e}, using identity mesh")
            return self.create_identity_mesh()
        
        return self.create_identity_mesh()
    
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
        
        # Create resizable window
        display = (self.window_width, self.window_height)
        pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
        pygame.display.set_caption("Real-time Mesh Warping - Press F11 for Fullscreen")
        
        # Set up orthographic projection
        self.setup_viewport(self.window_width, self.window_height)
        
        # Enable texturing
        glEnable(GL_TEXTURE_2D)
        
        # Create texture
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    
    def setup_viewport(self, width, height):
        """Setup viewport and projection for given window size"""
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.fullscreen = not self.fullscreen
        
        if self.fullscreen:
            # Get display info for fullscreen resolution
            info = pygame.display.Info()
            display = (info.current_w, info.current_h)
            pygame.display.set_mode(display, DOUBLEBUF | OPENGL | FULLSCREEN)
            self.window_width = info.current_w
            self.window_height = info.current_h
        else:
            # Return to windowed mode
            display = (800, 600)
            pygame.display.set_mode(display, DOUBLEBUF | OPENGL | RESIZABLE)
            self.window_width = 800
            self.window_height = 600
        
        self.setup_viewport(self.window_width, self.window_height)
        
    def handle_resize(self, width, height):
        """Handle window resize"""
        self.window_width = width
        self.window_height = height
        self.setup_viewport(width, height)
        
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
                if idx1 < len(self.mesh):
                    m1 = self.mesh[idx1]
                    glColor3f(m1['i'], m1['i'], m1['i'])
                    glTexCoord2f(m1['u'], m1['v'])
                    glVertex2f(m1['x'], m1['y'])
                
                # Top vertex
                idx2 = (r + 1) * self.cols + c
                if idx2 < len(self.mesh):
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
        if self.mesh_file:
            self.mesh = self.load_mesh(self.mesh_file)
            print("Reloaded mesh from file")
        else:
            self.mesh = self.create_identity_mesh()
            print("Reset to identity mesh")
    
    def run(self):
        """Main loop"""
        clock = pygame.time.Clock()
        running = True
        
        print("\nControls:")
        print("  F11/F - Toggle fullscreen")
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
                    elif event.key == K_F11 or event.key == K_f:
                        self.toggle_fullscreen()
                        print("Fullscreen:" if self.fullscreen else "Windowed mode")
                    elif event.key == K_b:
                        self.apply_barrel_distortion(0.3)
                        print("Applied barrel distortion")
                    elif event.key == K_p:
                        self.apply_pincushion_distortion(0.3)
                        print("Applied pincushion distortion")
                    elif event.key == K_r:
                        self.reset_mesh()
                    elif event.key == K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Resumed")
                elif event.type == VIDEORESIZE:
                    self.handle_resize(event.w, event.h)
            
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


class WarpingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Video Mesh Warping - Configuration")
        self.root.geometry("600x300")
        self.root.resizable(False, False)
        
        # Variables
        self.mesh_file = tk.StringVar()
        self.video_file = tk.StringVar()
        self.rows = tk.IntVar(value=20)
        self.cols = tk.IntVar(value=20)
        self.use_webcam = tk.BooleanVar(value=False)
        
        self.warper = None
        self.warper_thread = None
        
        self.create_widgets()
        
    def create_widgets(self):
        """Create GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title = ttk.Label(main_frame, text="Video Mesh Warping", 
                         font=("Arial", 16, "bold"))
        title.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Mesh file selection
        ttk.Label(main_frame, text="Mesh Map File:").grid(row=1, column=0, 
                                                           sticky=tk.W, pady=5)
        mesh_entry = ttk.Entry(main_frame, textvariable=self.mesh_file, width=40)
        mesh_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse...", 
                  command=self.browse_mesh).grid(row=1, column=2, pady=5)
        
        # Video file selection
        ttk.Label(main_frame, text="Video File:").grid(row=2, column=0, 
                                                        sticky=tk.W, pady=5)
        video_entry = ttk.Entry(main_frame, textvariable=self.video_file, width=40)
        video_entry.grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(main_frame, text="Browse...", 
                  command=self.browse_video).grid(row=2, column=2, pady=5)
        
        # Webcam checkbox
        webcam_check = ttk.Checkbutton(main_frame, text="Use Webcam (ignore video file)", 
                                       variable=self.use_webcam,
                                       command=self.toggle_webcam)
        webcam_check.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Mesh dimensions
        dims_frame = ttk.LabelFrame(main_frame, text="Mesh Dimensions (if no file)", 
                                    padding="10")
        dims_frame.grid(row=4, column=0, columnspan=3, pady=15, sticky=(tk.W, tk.E))
        
        ttk.Label(dims_frame, text="Rows:").grid(row=0, column=0, padx=5)
        ttk.Spinbox(dims_frame, from_=5, to=100, textvariable=self.rows, 
                   width=10).grid(row=0, column=1, padx=5)
        
        ttk.Label(dims_frame, text="Columns:").grid(row=0, column=2, padx=5)
        ttk.Spinbox(dims_frame, from_=5, to=100, textvariable=self.cols, 
                   width=10).grid(row=0, column=3, padx=5)
        
        # Start button
        start_btn = ttk.Button(main_frame, text="Start Warping", 
                              command=self.start_warping, 
                              style="Accent.TButton")
        start_btn.grid(row=5, column=0, columnspan=3, pady=20, ipadx=20, ipady=5)
        
        # Instructions
        info_frame = ttk.LabelFrame(main_frame, text="Instructions", padding="10")
        info_frame.grid(row=6, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        instructions = (
            "1. Select a mesh map file (.map) or leave empty for identity mesh\n"
            "2. Select a video file or check 'Use Webcam'\n"
            "3. Click 'Start Warping' to open the output window\n"
            "4. Press F11 or F in the output window for fullscreen"
        )
        ttk.Label(info_frame, text=instructions, justify=tk.LEFT).grid(row=0, column=0)
        
    def browse_mesh(self):
        """Browse for mesh file"""
        filename = filedialog.askopenfilename(
            title="Select Mesh Map File",
            filetypes=[
                ("Map files", "*.map"),
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.mesh_file.set(filename)
    
    def browse_video(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.video_file.set(filename)
    
    def toggle_webcam(self):
        """Handle webcam checkbox toggle"""
        # Could disable video file entry when webcam is selected
        pass
    
    def start_warping(self):
        """Start the warping process"""
        # Validate inputs
        if self.use_webcam.get():
            video_source = 0
        else:
            video_source = self.video_file.get()
            if not video_source or not os.path.exists(video_source):
                messagebox.showerror("Error", "Please select a valid video file or use webcam")
                return
        
        mesh_file = self.mesh_file.get() if self.mesh_file.get() else None
        
        # Start warping in separate thread
        def run_warper():
            try:
                self.warper = MeshWarper(
                    video_source, 
                    mesh_file, 
                    rows=self.rows.get(), 
                    cols=self.cols.get()
                )
                self.warper.run()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start warping: {str(e)}")
        
        self.warper_thread = threading.Thread(target=run_warper, daemon=True)
        self.warper_thread.start()
        
        # Minimize the GUI window
        self.root.iconify()
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()


def main():
    # Create and run GUI
    gui = WarpingGUI()
    gui.run()


if __name__ == "__main__":
    main()
