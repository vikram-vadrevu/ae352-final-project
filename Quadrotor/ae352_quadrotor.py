"""
This modules provides a backend for the ae352 quadrotor example
"""

###############################################################################
#DEPENDENCIES
###############################################################################
from condynsate.simulator import Simulator
from pathlib import Path
import numpy as np
import os


###############################################################################
#POSITION IN COURSE FUNCTIONS AND CLASS
###############################################################################
def get_gate_cen(gate, sim, scale):
    # Get the stl that was used to build the gate
    stl_pth = sim._get_urdf_vis_dat(gate)[0][0]
    stl_name = os.path.basename(stl_pth)
    
    # Extract gate height data based on gate size
    gate_height = 0.0
    if stl_name == 'gate_short.stl':
        gate_height=0.5
    elif stl_name == 'gate_med.stl':
        gate_height=1.5
    elif stl_name == 'gate_tall.stl':
        gate_height=2.0
        
    # Get the position of the base of the gate
    state = sim.get_base_state(urdf_obj=gate,
                               body_coords=False)
    pos = state['position']
    
    # Get the center of the gate
    cen = np.array([pos[0], pos[1], gate_height*scale])
    return cen
    
    
def get_gate_dirn(gate, sim):
    # Extract the state data
    state = sim.get_base_state(urdf_obj=gate,
                               body_coords=False)
    
    # Apply the yaw rotation to the base pointing direction
    yaw = state['yaw']
    c = np.cos(yaw)
    s = np.sin(yaw)
    Rz = np.array([[c, -s, 0.],
                   [s,  c, 0.],
                   [0., 0., 1.]])
    base = np.array([0., 1., 0.])
    dirn = Rz @ base
    
    # Return the calculated direction
    return dirn

    
class Course_Tracker():
    def __init__(self, quadrotor, sim, gate_data):
        # Track relevant data
        self.quadrotor = quadrotor
        self.sim  = sim
        self.gate_data = gate_data
        
        # Get the state of the quadrotor in world coords
        world_state = self.sim.get_base_state(urdf_obj=self.quadrotor,
                                              body_coords=False)
        position = world_state['position']
                
        # Set the current gate and side
        self.num_laps = 0
        self.curr_gate = 0
        self.start_side = self.get_gate_side(position)
        
        # Get the time tracking variables
        self.lap_start_time = None
        
        
    def reset(self):
        # Get the state of the quadrotor in world coords
        world_state = self.sim.get_base_state(urdf_obj=self.quadrotor,
                                              body_coords=False)
        position = world_state['position']
        
        # Reset current gate and current gate side
        self.num_laps = 0
        self.curr_gate = 0
        self.start_side  = self.get_gate_side(position)

        # Get the time tracking variables
        self.lap_start_time = None


    def near_gate(self, position):
        # Extract gate data
        gate_cen = self.gate_data['cens'][self.curr_gate]
        gate_radius = self.gate_data['radii'][self.curr_gate]
        
        # Get the distance from the quadrotor to the gate center
        delta_to_gate = gate_cen - position
        dist = np.linalg.norm(delta_to_gate)
        
        # Return if the distance from the quadrotor to gate center is less than
        # the radius of the gate
        return dist < gate_radius
    
    
    def get_gate_side(self, position):
        # Extract gate data
        gate_cen = self.gate_data['cens'][self.curr_gate]
        gate_dirn = self.gate_data['dirns'][self.curr_gate]
        
        # Get the dot product of the gate direction and direction from
        # quadrotor to gate center
        delta_to_gate = gate_cen - position
        dotp = np.dot(delta_to_gate, gate_dirn)
        
        # Get the side of the gate the quadrotor is on. 1 is before the gate,
        # -1 is after the gate.
        side = int(dotp / abs(dotp))
        return side
    
    
    def passed_gate(self, position):
        # Determine the current side of the gate and whether or not the
        # quadrotor is near the gate
        in_gate_radius = self.near_gate(position)
        curr_side = self.get_gate_side(position)
        
        # If we are near the gate and we change side of the gate, we passed 
        # through it
        if curr_side != self.start_side and in_gate_radius:
            passed = True
        else:
            passed = False
            
        # Return results
        return passed
    
    
    def step(self, position, verbose):
        # Set the lap start time
        if self.lap_start_time == None:
            self.lap_start_time = self.sim.time
            
        # If the current gate is passed...
        if self.passed_gate(position):
            # Get elapsed time
            elapsed = self.sim.time - self.lap_start_time
            elapsed = np.round(elapsed, 3)
            
            # Print to screen...
            if verbose:
                print("PASSED GATE {} IN {}s".format(self.curr_gate+1,
                                                     elapsed))
            
            # update the current gate and lap number...
            self.curr_gate = self.curr_gate + 1
            if self.curr_gate >= len(self.gate_data['cens']):
                self.curr_gate = 0
                self.num_laps = self.num_laps + 1
                if verbose:
                    print("COMPLETED LAP IN {}s".format(elapsed))
                self.lap_start_time = self.sim.time
                
            # and update the side on which we start.
            self.start_side = self.get_gate_side(position)
            
        # Return the current gate's position direction and if it's the last one
        cen = self.gate_data['cens'][self.curr_gate]
        dirn = self.gate_data['dirns'][self.curr_gate]
        is_last = (self.curr_gate == len(self.gate_data['cens']) - 1)
        return cen, dirn, is_last, self.num_laps
            
    
###############################################################################
#SIMULATION CLASS
###############################################################################
class Quadrotor_Sim():
    def __init__(self,
                 use_keyboard=True,
                 visualization=True,
                 visualization_fr=60.,
                 animation=True,
                 animation_fr=10.,
                 verbose=False):
        """
        Initializes an instance of the simulation class.

        Parameters
        ----------
        use_keyboard : bool, optional
            A boolean flag that indicates whether the simulation will allow
            the use of keyboard interactivity. The default is True.
        visualization : bool, optional
            A boolean flag that indicates whether the simulation will be 
            visualized in meshcat. The default is True.
        visualization_fr : float, optional
            The frame rate (frames per second) at which the visualizer is
            updated. The default is 40..
        animation : bool, optional
            A boolean flag that indicates whether animated plots are created
            in real time. The default is True.
        animation_fr : float, optional
            The frame rate (frames per second) at which the animated plots are
            updated. The default is 10..
        verbose : Bool, optional
            A boolean flag that indicates whether warnings are printing during
            function execution. The default is False.
            
        Returns
        -------
        None.

        """
        # Check all inputs
        if not isinstance(use_keyboard, bool):
            if verbose:
                string = "use_keyboard must be type {}."
                print(string.format(bool))
            return None
        if not isinstance(visualization, bool):
            if verbose:
                string = "visualization must be type {}."
                print(string.format(bool))
            return None
        if not isinstance(animation, bool):
            if verbose:
                string = "animation must be type {}."
                print(string.format(bool))
            return None
        if not isinstance(visualization_fr, float):
            if verbose:
                string = "visualization_fr must be type {}."
                print(string.format(float))
            return None
        if not isinstance(animation_fr, float):
            if verbose:
                string = "animation_fr must be type {}."
                print(string.format(float))
            return None
        
        # Set the visualization and animation options
        self.visualization = visualization
        self.animation = animation

        # Initialize and instance of the simulator
        self.sim = Simulator(keyboard=use_keyboard,
                             visualization=visualization,
                             visualization_fr=visualization_fr,
                             animation=animation,
                             animation_fr=animation_fr)
        
        # Get the path to the current directory
        path = (Path(__file__).parents[0]).absolute().as_posix()
        
        # Load the quadrotor
        urdf_path = path + "/quadcopter_vis/flagless_quadcopter.urdf"
        self.quad_obj = self.sim.load_urdf(urdf_path=urdf_path,
                                           position=[0., 0., 0.],
                                           fixed=False,
                                          update_vis=True)
            
        # Load the ground
        urdf_path = path + "/quadcopter_vis/plane.urdf"
        tex_path = path + "/quadcopter_vis/carpet.png"
        _ = self.sim.load_urdf(urdf_path=urdf_path,
                               tex_path=tex_path,
                               position=[0., 0., -0.051],
                               fixed=True,
                               update_vis=False)

        # Define the course
        gate_scale = 5.
        pos_scale = 0.6
        p1 = pos_scale*np.array([20., 0., 0.])
        p2 = pos_scale*np.array([35., 13., 0.])
        p3 = pos_scale*np.array([40., 30., 0.])
        p4 = pos_scale*np.array([20., 45., 0.])
        p5 = pos_scale*np.array([0., 30., 0.])
        p6 = pos_scale*np.array([0., -16., 0.])
        p7 = pos_scale*np.array([-6., -22., 0.])
        y1 = -np.pi/2.
        y2 = -np.pi/6.
        y3 = 0.
        y4 = np.pi/2.
        y5 = np.pi
        y6 = np.pi
        y7 = np.pi/2.

        # Load the course
        urdf_path = path + "/quadcopter_vis/gate_short.urdf"
        gate_1 = self.sim.load_urdf(urdf_path=urdf_path,
                                    position=p1,
                                    yaw=y1,
                                    fixed=True,
                                    update_vis=False)
        urdf_path = path + "/quadcopter_vis/gate_short.urdf"
        gate_2 = self.sim.load_urdf(urdf_path=urdf_path,
                                    position=p2,
                                    yaw=y2,
                                    fixed=True,
                                    update_vis=False)
        urdf_path = path + "/quadcopter_vis/gate_med.urdf"
        gate_3 = self.sim.load_urdf(urdf_path=urdf_path,
                                    position=p3,
                                    yaw=y3,
                                    fixed=True,
                                    update_vis=False)
        urdf_path = path + "/quadcopter_vis/gate_tall.urdf"
        gate_4 = self.sim.load_urdf(urdf_path=urdf_path,
                                    position=p4,
                                    yaw=y4,
                                    fixed=True,
                                    update_vis=False)
        urdf_path = path + "/quadcopter_vis/gate_tall.urdf"
        gate_5 = self.sim.load_urdf(urdf_path=urdf_path,
                                    position=p5,
                                    yaw=y5,
                                    fixed=True,
                                    update_vis=False)
        urdf_path = path + "/quadcopter_vis/gate_tall.urdf"
        gate_6 = self.sim.load_urdf(urdf_path=urdf_path,
                                    position=p6,
                                    yaw=y6,
                                    fixed=True,
                                    update_vis=False)
        urdf_path = path + "/quadcopter_vis/gate_med.urdf"
        gate_7 = self.sim.load_urdf(urdf_path=urdf_path,
                                    position=p7,
                                    yaw=y7,
                                    fixed=True,
                                    update_vis=False)
        
        # Compile the gate data
        gates = [gate_1, gate_2, gate_3, gate_4, gate_5, gate_6, gate_7]
        self.gate_data = {"gates" : gates,
                          "cens" : [],
                          "dirns" : [],
                          "radii" : []}
        for gate in self.gate_data['gates']:
            cen = get_gate_cen(gate, self.sim, gate_scale)
            dirn = get_gate_dirn(gate, self.sim)
            self.gate_data['cens'].append(cen)
            self.gate_data['dirns'].append(dirn)
            self.gate_data['radii'].append(0.5*gate_scale)
        
        # Make a course tracker(s)
        self.tracker = Course_Tracker(self.quad_obj,
                                 self.sim,
                                 self.gate_data)
        
        # Only set the background lighting if the visualizer is being used
        if self.visualization:
            self.sim.set_background(top_color=[43, 50, 140],
                                    bot_color=[125, 149, 219])
            self.sim.set_posx_pt_light(on=True,
                                       intensity=0.5,
                                       distance=10.)
            self.sim.set_negx_pt_light(on=True,
                                       intensity=0.5,
                                       distance=10.)
            self.sim.set_ambient_light(on=True,
                                       intensity=0.65)
        
        # If there is animation, add subplots
        if self.animation:            
            # Make plot for state and input
            self.p1, self.a1 = self.sim.add_subplot(
                                    n_artists=1,
                                    subplot_type='line',
                                    title="Ground Track",
                                    x_label="x position [m]",
                                    y_label="y position [m]",
                                    colors=["k"],
                                    line_widths=[2.5],
                                    line_styles=["-"],
                                    h_zero_line=True,
                                    v_zero_line=True)
            
            # Open the animator GUI
            self.sim.open_animator_gui()


    def run(self,
            controller,
            collect_data = True,
            max_time = None,
            verbose=False):
        """
        Runs a complete simulation

        Parameters
        ----------
        controller : Controller class
            A member of a custom class that, at a minimum, must provide the
            functions controller.run() and controller.reset()
        collect_data : bool, optional
            A boolean flag that indicates whether or not simulation data is
            collected
        max_time : Float, optional
            The total amount of time the simulation is allowed to run. When
            set to None, the simulator will run until the terminal command is 
            called. If the keyboard is disabled, users are not allowed to 
            set max time as None. The default value is None. 
        verbose : Bool, optional
            A boolean flag that indicates whether warnings are printing during
            function execution. The default is False.

        Returns
        -------
        data : Dictionary of Lists
            data["position"] : List of arrays, shape(n,3)
                The [x, y, z] position of the quadrotor at each time stamp.
            data["orientation"] : List of arrays, shape(n,3)
                The [roll, pitch, yaw] orientation of the quadrotor at each
                time stamp.
            data["velocity"] : List of arrays, shape(n,3)
                The [vx, vy, vz] velocity of the quadrotor in body fixed 
                coordinates at each time stamp.
            data["angular velocity"] : List of arrays, shape(n,3)
                The [wx, wy, wz] angular velocity of the quadrotor in body 
                fixed coordinates at each time stamp.
            data["inputs"] : list of arrays, shape(n,4):
                The inputs applied to the quadrotor. [tau_x, tau_y, tau_z, f_z]
            data["time"] : List of Floats, shape(n,)
                A list of the time stamps in seconds.

        """
        # Check all inputs
        if (not isinstance(max_time, float)) and (not max_time == None):
            if verbose:    
                string = "max_time must be type {} or {}."
                print(string.format(None, float))
            return None
        
        # Reset the simulator and tracker
        self.sim.reset()
        self.tracker.reset()

        # Reset the controller
        self.controller = controller
        self.controller.reset()
        
        # Create a dictionary to hold the simulation data
        self.collect_data = collect_data
        if self.collect_data:
            self.data = {"position" : [],
                         "orientation" : [],
                         "velocity" : [],
                         "angular velocity" : [],
                         "inputs" : [],
                         "time" : []}

        # Await run command if visualization or animation is on
        if self.visualization or self.animation:
            self.sim.await_keypress(key='enter')

        # Run the simulation loop
        while not self.sim.is_done:      
            ###############################################################
            # SENSORS
            # Get the state of the quadrotor in world coords
            world_state = self.sim.get_base_state(urdf_obj=self.quad_obj,
                                                  body_coords=False)
            position = world_state['position']
            orientation = np.array([world_state['roll'],
                                    world_state['pitch'],
                                    world_state['yaw']])
            
            # Get the state of the quadrotor in body coords
            body_state = self.sim.get_base_state(urdf_obj=self.quad_obj,
                                                 body_coords=True)
            vel = body_state['velocity']
            ang_vel = body_state['angular velocity']

            ###############################################################
            # COURSE DETECTION
            # Determine where the quadrotor is
            _, _, _, _ = self.tracker.step(position, verbose)
                
            ###############################################################
            # CONTROLLER
            # Determine keypresses
            key_w = self.sim.is_pressed("w")
            key_s = self.sim.is_pressed("s")
            key_q = self.sim.is_pressed("q")
            key_e = self.sim.is_pressed("e")
            key_i = self.sim.is_pressed("i")
            key_k = self.sim.is_pressed("k")
            key_j = self.sim.is_pressed("j")
            key_l = self.sim.is_pressed("l")
            
            # Get the inputs for controller i
            inputs = self.controller.run(dt = self.sim.dt,
                                         time = self.sim.time,
                                         key_w=key_w,
                                         key_s=key_s,
                                         key_q=key_q,
                                         key_e=key_e,
                                         key_i=key_i,
                                         key_k=key_k,
                                         key_j=key_j,
                                         key_l=key_l)
        
            # Extract the inputs
            tau_x = inputs[0]
            tau_y = inputs[1]
            tau_z = inputs[2]
            f_z = inputs[3]
            
            # Limit the inputs
            tau_x = np.clip(tau_x, -0.03, 0.03)
            tau_y = np.clip(tau_y, -0.03, 0.03)
            tau_z = np.clip(tau_z, -0.03, 0.03)
            f_z = np.clip(f_z, 0.0, 14.72)
            
            # Make sure valid
            vx=isinstance(tau_x,(int,float)) and not isinstance(tau_x,bool)
            vy=isinstance(tau_y,(int,float)) and not isinstance(tau_y,bool)
            vz=isinstance(tau_z,(int,float)) and not isinstance(tau_z,bool)
            vfz=isinstance(f_z,(int,float)) and not isinstance(f_z,bool)
            if not vx or (tau_x!=tau_x):
                tau_x = 0.0
                if verbose:
                    string = "tau_x must be type {} or {}."
                    print(string.format(int, float))
            if not vy or (tau_y!=tau_y):
                tau_y = 0.0
                if verbose:
                    string = "tau_y must be type {} or {}."
                    print(string.format(int, float))
            if not vz or (tau_z!=tau_z):
                tau_z = 0.0
                if verbose:
                    string = "tau_z must be type {} or {}."
                    print(string.format(int, float))
            if not vfz or (f_z!=f_z):
                f_z = 0.0
                if verbose:
                    string = "f_z must be type {} or {}."
                    print(string.format(int, float))
            
            ###############################################################
            # ACTUATOR
            # Apply the inputs
            self.sim.apply_force_to_com(urdf_obj=self.quad_obj,
                                        force=[0,0,f_z],
                                        body_coords=True,
                                        show_arrow=self.visualization,
                                        arrow_scale=0.05,
                                        arrow_offset=0.0)
            self.sim.apply_external_torque(urdf_obj=self.quad_obj,
                                           torque=[tau_x, tau_y, tau_z],
                                           body_coords=True,
                                           show_arrow=self.visualization,
                                           arrow_scale=10.0,
                                           arrow_offset=0.)
            
            ###############################################################
            # SIMULATION DATA
            # Append data to history lists
            if self.collect_data:
                self.data["position"].append(position)
                self.data["orientation"].append(orientation)
                self.data["velocity"].append(vel)
                self.data["angular velocity"].append(ang_vel)
                self.data["inputs"].append([tau_x, tau_y, tau_z, f_z])
                self.data["time"].append(self.sim.time)
            
            ###############################################################
            # UPDATE THE PLOTS
            # This is how we add data points to the animator
            if self.animation:
                self.sim.add_subplot_point(subplot_index=self.p1,
                                           artist_index=self.a1[0],
                                           x=position[0],
                                           y=position[1])
            
            ###################################################################
            # STEP THE SIMULATION
            # Step the sim
            val = self.sim.step(real_time=self.visualization or self.animation,
                                update_vis=self.visualization,
                                update_ani=self.animation,
                                max_time=max_time)
            
            # Handle resetting the controller and simulation data
            if val == 3:
                
                # Reset the controller and tracker
                self.controller.reset()
                self.tracker.reset()
                        
                # Reset the history
                if self.collect_data:
                    self.data = {"position" : [],
                                 "orientation" : [],
                                 "velocity" : [],
                                 "angular velocity" : [],
                                 "inputs" : [],
                                 "time" : []}
       
        # When the simulation is done running, return the data
        if self.collect_data:
            return self.data
        else:
            return None
