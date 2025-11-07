import mujoco
import mujoco_viewer
from snn_model import Spiking_Network, Spike_Act_with_Surrogate_Gradient
import numpy as np
import pickle
import time
import cv2
from collections import deque
import imageio
from scipy.spatial.transform import Rotation as R
from behavior import follow, docking, inverse_docking
from controller import rotate, STAController
# from inverse_dynamic import InverseDynamicsModel, MassIdfModel
import torch
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--time_steps', default=20, type=int)
parser.add_argument('--channel', default=100, type=int)
parser.add_argument('--epoch', default=500, type=int)

parser.add_argument('--vth', default=0.4, type=float)
parser.add_argument('--decay', default=0.1, type=float)
args = parser.parse_args()

def delta_theta(theta, theta_prev):
    delta = theta - theta_prev
    return (delta + np.pi) % (2 * np.pi) - np.pi  # Wrap to (-pi, pi]
class vehicle():
    def __init__(self, name, x=0, y=0, z=0, theta=0):
        self.name = name
        self.x = x
        self.y = y
        self.z = z
        self.theta = theta
        self.prev_x = x
        self.prev_y = y
        self.prev_theta = theta
        d=0.3
    def update(self, x=None, y=None, z=None, theta=None):
        # Store previous values
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_z = self.z
        self.prev_theta = self.theta
        if x:
            self.x = x 
        if y:

            self.y = y 
        if z:
            self.z = z 
        if theta:
            self.theta = theta 

        self.rm_x1 = self.x - 0.2*np.cos(self.theta)
        self.rm_y1 = self.y - 0.2*np.sin(self.theta)
        self.rm_x2 = self.x + 0.2*np.cos(self.theta)
        self.rm_y2 = self.y + 0.2*np.sin(self.theta)
        self.plane_x = self.rm_x1 + 0.5*np.sin(self.theta)
        self.plane_y = self.rm_y1 - 0.5*np.cos(self.theta)
        self.out_x = self.rm_x1 + np.sin(self.theta)
        self.out_y = self.rm_y1 - np.cos(self.theta)
        # Update with new values (only if not None)
        

def update(vehicles, idxs, data):
    for i in range(len(vehicles)):
        vh = vehicles[i]
        idx = idxs[i]
        x, y, z = data.xpos[idx][0],data.xpos[idx][1],data.xpos[idx][2]
        r = R.from_quat(data.xquat[idx])
        euler = r.as_euler('zxy')
        if vh.name == 'ap':
            theta = -euler[-1] + np.pi
        else:
            
            theta = euler[1]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        vh.update(x, y, z, theta)
def init_control_param(rm1, rm2):
    return rm1.x, rm1.y, rm1.z, rm1.theta, rm2.x, rm2.y, rm2.z, rm2.theta

def mecanum_wheel_velocities(vx, vy, w):
    r = 0.05 
    Lx = 0.3
    Ly = 0.2
    R = Lx + Ly

    # Compute angular velocities
    omega_front_left = (1 / r) * (vx - vy - R * w)
    omega_front_right = (1 / r) * (vx + vy + R * w)
    omega_rear_left = (1 / r) * (vx + vy - R * w)
    omega_rear_right = (1 / r) * (vx - vy + R * w)

    return omega_front_left, omega_front_right, omega_rear_left, omega_rear_right



if __name__=="__main__":
    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path("./AllScene/Scene.xml")
    data = mujoco.MjData(model)
    
    viewer = mujoco_viewer.MujocoViewer(model, data, 'offscreen')
    # viewer = mujoco_viewer.MujocoViewer(model, data)
    # controller = STAController(c=0.5, k1_x=30, k2_x=0, kd_x = 0.001/3, k1_y = 0.4, kd_y=0.2/3)
    controller = STAController(c=0.5, k1_x=30*1.1, k2_x=0, kd_x = 0.001, k1_y = 0.4*1.1, kd_y=0.2)

    # ffc = Spiking_Network(10, hidden_dim=[64, 64], output_dim=2,
    #                         num_channels=args.channel, vth=args.vth, decay=args.decay, activation=Spike_Act_with_Surrogate_Gradient.apply, on_apu=False, init_batch=args.batch_size).to(device)
    # ffc.load_state_dict(torch.load('./models/IDM/best_model.pkl', weights_only=True))
    # pe = Spiking_Network(11, hidden_dim=[64, 64], output_dim=1,
    #                         num_channels=args.channel, vth=args.vth, decay=args.decay, activation=Spike_Act_with_Surrogate_Gradient.apply, on_apu=False, init_batch=args.batch_size).to(device)
    # pe.load_state_dict(torch.load('./models/IDM/best_model.pkl', weights_only=True))
    # ffc.eval()
    # pe.eval()
    control_step = 20
    fps = int(1 / model.opt.timestep/control_step)  # Get the simulation FPS
    iteration = 0

    # n_w
    seq_len = 20
    state_queue = deque(maxlen=seq_len)


    rm1_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'rm_base')
    rm2_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'rm_base_2')
    wheel_joints = ['rm_front_left_wheel_joint', 'rm_front_right_wheel_joint', 'rm_rear_left_wheel_joint', 'rm_rear_right_wheel_joint']
    rm1_wheel_joint_id = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, wheel_joints[i])] for i in range(4)]
    rm1_wheel_joint_id = np.array(rm1_wheel_joint_id)-1
    rm2_wheel_joint_id = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, wheel_joints[i]+'_2')] for i in range(4)]
    rm2_wheel_joint_id = np.array(rm2_wheel_joint_id)-1
    # apollo_base_id = model.body("apollo_base").id
    apollo_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'apollo_base')
    # Get the position of the body (in world coordinates)
    velocity_prev = np.zeros(2)
    acceleration_prev = np.zeros(2)
    t = 0
    N = 1000
    with open('desired_trajectory/desired_pickup.pkl','rb') as f:
        desired_pickup = np.array(pickle.load(f))
    frames_ap = []
    frames_rm1 = []
    frames_rm2 = []
    delay_time = 1
    delay_steps = int(delay_time/model.opt.timestep/control_step)
    desired_states = []
    desired_velocitys = []
    desired_accelerations = []
    for phase in range(4):
        with open(f'desired_trajectory/apollo_phase_{phase}.pkl','rb') as f:
            desired_coords = np.array(pickle.load(f))
        desired_velocitys.append(np.diff(desired_coords,axis=0)/model.opt.timestep/control_step)
        desired_accelerations.append(np.diff(desired_velocitys[-1],axis=0)/model.opt.timestep/control_step)
        desired_states.append(desired_coords[:-2])

    phase = 0
    mujoco.mj_step(model, data)
    ap = vehicle('ap')
    rm1 = vehicle('rm1')
    rm1_carried = True
    rm2 = vehicle('rm2')
    rm2_carried = True
    update([ap, rm1, rm2], [apollo_base_id, rm1_id, rm2_id], data)
    rm1_x, rm1_y, rm1_z, rm1_theta, rm2_x, rm2_y, rm2_z, rm2_theta = init_control_param(rm1, rm2)
    height = 0
    angle = 0
    dock_id = 0
    # while True:

    prev_ffc_throttle = -2/3
    prev_ffc_steer = 0
    steer_angle = 0
    t1 = time.time()

    ## addtional mass
    mass_idx = -1
    # mass_idx = 24
    model.body_mass[mass_idx] += 400

    predict_m = 0
    prev_desired_velocity_rot = np.zeros(2)
    prev_desired_acceleration_rot = np.zeros(2)
    actual_pickup = []
    stuck_steps = 0
    throttle_value = 0
    dx = 0
    dy = 0
    while True:
        e_lon , e_lat=None, None
        noise = 0.2
        vx, vy = data.qvel[0], data.qvel[1]
        current_state = np.array([ap.x + noise * (np.random.random() - 0.5), ap.y + noise * (np.random.random() - 0.5), ap.theta])
        current_velocity = np.array([vx, vy])

        if phase>=0:
            desired_state = desired_states[phase][iteration]
            desired_velocity = desired_velocitys[phase][iteration]
            desired_acceleration = desired_accelerations[phase][iteration]
            desired_velocity_rot = rotate(desired_velocity, ap.theta)
            desired_acceleration_rot = rotate(desired_acceleration, ap.theta)
            dtheta = delta_theta(ap.theta, ap.prev_theta)
        else:
            desired_state = desired_pickup[-phase-1].copy()

        if phase==0:
            dist = np.linalg.norm(desired_pickup[phase]-np.array([ap.x, ap.y]))
            # if dist < 0.2 and np.linalg.norm([vx, vy]) < 0.1:
            if dist < 0.5:
                actual_pickup.append([ap.x, ap.y])
                rm1_carried = False
                iteration = 0
                phase = -1
                stuck_steps = 0
                print('Entering phase -1: unload rm1')

            smc_throttle, smc_steer, e_lon , e_lat= controller.compute_controls(
                current_state, current_velocity, desired_state, desired_velocity
            )
            
            desired_obs = np.concatenate([desired_velocity_rot, desired_acceleration_rot, prev_desired_velocity_rot, prev_desired_acceleration_rot, [dtheta, predict_m]], dtype=np.float32)
            # [ffc_throttle, ffc_steer] = ffc.forward(PC(desired_obs)).detach().numpy()
            prev_desired_velocity_rot = desired_velocity_rot.copy()
            prev_desired_acceleration_rot = desired_acceleration_rot.copy()
            
            rm1_x, rm1_y, rm1_theta, rm2_x, rm2_y, rm2_theta = follow(ap)
            if iteration<len(desired_states[phase])-1:
                iteration+=1
                
            else:
                stuck_steps += 1
                print(phase, dist, np.linalg.norm([vx, vy]))
            if stuck_steps>50:
                break
        ## unload rm 1
        elif phase==-1:
            smc_steer = -steer_angle
            smc_throttle = 0
            ffc_steer = 0
            ffc_throttle = -2/3
            args, finished = docking(dock_id, ap)
            state_queue.clear()
            if finished:
                # predict_m = 0
                prev_desired_velocity_rot = np.zeros(2)
                prev_desired_acceleration_rot = np.zeros(2)
                model.body_mass[mass_idx] -= 100
                phase = 1
                dock_id = 0
                print('Entering phase 1')
            else:
                height, angle, rm1_x, rm1_y, rm1_z, rm1_theta, dock_id = args
                _, _, _, rm2_x, rm2_y, rm2_theta = follow(ap)
        elif phase==1:
            dist = np.linalg.norm(desired_pickup[phase]-np.array([ap.x, ap.y]))
            # if dist < 0.2 and np.linalg.norm([vx, vy]) < 0.1:
            if dist < 0.5:
                rm2_carried = False
                actual_pickup.append([ap.x, ap.y])
                stuck_steps = 0
                iteration = 0
                phase = -2
                print('Entering phase -2: unload rm2')
            
            smc_throttle, smc_steer, e_lon , e_lat= controller.compute_controls(
                current_state, current_velocity, desired_state, desired_velocity
            )
            
            desired_obs = np.concatenate([desired_velocity_rot, desired_acceleration_rot, prev_desired_velocity_rot, prev_desired_acceleration_rot, [dtheta, predict_m]], dtype=np.float32)

            # [ffc_throttle, ffc_steer] = ffc.forward(PC(TCE(desired_obs))).detach().numpy()
            prev_desired_velocity_rot = desired_velocity_rot.copy()
            prev_desired_acceleration_rot = desired_acceleration_rot.copy()
            _, _, _, rm2_x, rm2_y, rm2_theta = follow(ap)
            if iteration<len(desired_states[phase])-1:
                iteration+=1
                
            else:
                stuck_steps+=1
                print(phase, dist, np.linalg.norm([vx, vy]))
            if stuck_steps>50:
                break

        ## unload rm 2
        elif phase==-2:
            
            smc_steer = -steer_angle
            smc_throttle = 0
            ffc_steer = 0
            ffc_throttle = -2/3
            args, finished = docking(dock_id, ap, second=True)
            state_queue.clear()
            if finished:
                # predict_m = 0
                prev_desired_velocity_rot = np.zeros(2)
                prev_desired_acceleration_rot = np.zeros(2)
                model.body_mass[mass_idx] -= 100
                phase = 2
            
                print('Entering phase 2')
                dock_id = 0
            else:
                height, angle, rm2_x, rm2_y, rm2_z, rm2_theta, dock_id = args
        elif phase==2:
            rm1_x, rm1_y, rm1_theta = desired_pickup[phase][0] + 1, desired_pickup[phase][1] - 1, np.pi/2 * 1.1
            dist = np.linalg.norm(desired_pickup[phase]-np.array([ap.x, ap.y]))
            # if dist < 0.2 and np.linalg.norm([vx, vy]) < 0.1:
            if dist < 0.5:
                stuck_steps = 0
                actual_pickup.append([ap.x, ap.y])
                iteration = 0
                phase = -3
                print('Entering phase -3: load rm1')

            smc_throttle, smc_steer, e_lon , e_lat= controller.compute_controls(
                current_state, current_velocity, desired_state, desired_velocity
            )
            
            desired_obs = np.concatenate([desired_velocity_rot, desired_acceleration_rot, prev_desired_velocity_rot, prev_desired_acceleration_rot, [dtheta, predict_m]], dtype=np.float32)

            # [ffc_throttle, ffc_steer] = ffc.forward(PC(desired_obs)).detach().numpy()
            prev_desired_velocity_rot = desired_velocity_rot.copy()
            prev_desired_acceleration_rot = desired_acceleration_rot.copy()

            if iteration<len(desired_states[phase])-1:
                iteration+=1
                
            else:
                stuck_steps+=1
                print(phase, dist, np.linalg.norm([vx, vy]))
            if stuck_steps>50:
                break
        # load rm1
        elif phase==-3:
            smc_steer = -steer_angle
            smc_throttle = 0
            ffc_steer = 0
            ffc_throttle = -2/3
            args, finished = inverse_docking(dock_id, ap, rm1, second=True)
            state_queue.clear()
            if finished:
                # predict_m = 0
                prev_desired_velocity_rot = np.zeros(2)
                prev_desired_acceleration_rot = np.zeros(2)
                rm1_carried = True
                model.body_mass[mass_idx] += 100
                phase = 3
                print('Entering phase 3')
                dock_id = 0
            else:
                height, angle, rm1_x, rm1_y, rm1_z, rm1_theta, dock_id = args
        elif phase==3:
            rm2_x, rm2_y, rm2_theta = desired_pickup[phase][0]-1.7, desired_pickup[phase][1]+0.5, 0.2
            dist = np.linalg.norm(desired_pickup[phase]-np.array([ap.x, ap.y]))
            # if dist < 0.2 and np.linalg.norm([vx, vy]) < 0.1:
            if dist < 0.5:
                actual_pickup.append([ap.x, ap.y])
                stuck_steps = 0
                iteration = 0
                phase = -4
                print('Entering phase -4: load rm2')

            smc_throttle, smc_steer, e_lon , e_lat= controller.compute_controls(
                current_state, current_velocity, desired_state, desired_velocity
            )
            
            desired_obs = np.concatenate([desired_velocity_rot, desired_acceleration_rot, prev_desired_velocity_rot, prev_desired_acceleration_rot, [dtheta, predict_m]], dtype=np.float32)

            # [ffc_throttle, ffc_steer] = ffc.forward(PC(desired_obs)).detach().numpy()
            prev_desired_velocity_rot = desired_velocity_rot.copy()
            prev_desired_acceleration_rot = desired_acceleration_rot.copy()
            
            _, _, _, rm1_x, rm1_y, rm1_theta = follow(ap)
            if iteration<len(desired_states[phase])-1:
                iteration+=1
                
            else:
                stuck_steps+=1
                print(phase, dist, np.linalg.norm([vx, vy]))
            if stuck_steps>50:
                break
        # load rm 2
        elif phase==-4:
            smc_steer = -steer_angle
            smc_throttle = 0
            ffc_steer = 0
            ffc_throttle = -2/3
            args, finished = inverse_docking(dock_id, ap, rm2)
            state_queue.clear()
            if finished:
                # predict_m = 0
                prev_desired_velocity_rot = np.zeros(2)
                prev_desired_acceleration_rot = np.zeros(2)
                model.body_mass[mass_idx] += 100
                rm2_carried = True
                print('Ending')
                dock_id = 0
                break
            else:
                _, _, _, rm1_x, rm1_y, rm1_theta = follow(ap)
                height, angle, rm2_x, rm2_y, rm2_z, rm2_theta, dock_id = args

        
        # if len(state_queue) == seq_len:
        #     predict_m = np.mean(pe.forward(TCE(np.array(state_queue, dtype=np.float32)))).detach().numpy()
        #     predict_m = np.clip(predict_m, 0, 1)




        # ffc_throttle = prev_ffc_throttle * 0.9 + ffc_throttle * 0.1
        # prev_ffc_throttle = ffc_throttle
        # ffc_throttle = ffc_throttle*60 + 40         # rescale to 40-100

        
        # throttle_control = smc_throttle + ffc_throttle
        # steer_control = smc_steer + ffc_steer

        throttle_control = smc_throttle
        steer_control = smc_steer
        steer_control = np.clip(steer_control, -0.5/delay_steps, 0.5/delay_steps)
        # print('            STA              FFC')
        # print('Throttle:', smc_throttle, ffc_throttle)
        # print('Steer:', smc_steer, ffc_steer,'\n')

        steer_angle += steer_control
        steer_angle= np.clip(steer_angle, -0.5, 0.5)



        current_acceleration = (current_velocity - velocity_prev)/model.opt.timestep/control_step
        obs = np.concatenate([rotate(current_velocity,ap.theta), rotate(current_acceleration,ap.theta), velocity_prev, acceleration_prev, [dtheta, (throttle_control-40)/60, steer_angle*2]], dtype=np.float32)
        velocity_prev = current_velocity.copy()
        acceleration_prev = current_acceleration.copy()
        state_queue.append(obs)
        
        ## asign these variables to control!!!
        ## throttle, steer, height, angle
        ## rm1_x, rm1_y, rm1_z, rm1_theta
        ## rm2_x, rm2_y, rm2_z, rm2_theta
        # rm1_theta = np.pi/2
        # rm2_theta = -np.pi/2
        data.ctrl[0] = throttle_control  # Forward motor control
        data.ctrl[1] = data.ctrl[0]  # Forward motor control
        data.ctrl[2] = steer_angle    # Turning motor control
        data.ctrl[3] = data.ctrl[2]     # Turning motor control
        ### handle: 0, up, -0.63 down
        data.ctrl[4] = height
        ### platform: 0, up, 1.6 down
        data.ctrl[5] = angle
        # x1,y1,x2,y2 = follow(ap)

        qx = np.sin(-rm1_theta / 2-np.pi/2)
        qw = np.cos(-rm1_theta / 2-np.pi/2)
        data.mocap_pos[0] = np.array([rm1_x, rm1_y, rm1_z+0.01])
        data.mocap_quat[0] = np.array([qx, 0, 0, qw])
        # data.mocap_quat[0] = R.from_euler('zxy', [0, rm1_theta_temp, 0]).as_quat()

        qx = np.sin(-rm2_theta / 2-np.pi/2)
        qw = np.cos(-rm2_theta / 2-np.pi/2)
        data.mocap_pos[1] = np.array([rm2_x, rm2_y, rm2_z + 0.01])
        data.mocap_quat[1] = np.array([qx, 0, 0, qw])
        # data.mocap_quat[1] = R.from_euler('zxy', [0, rm2_theta_temp, 0]).as_quat()

        if not rm1_carried:
            rm1_vx = (rm1.x - rm1.prev_x)/model.opt.timestep/control_step
            rm1_vy = (rm1.y - rm1.prev_y)/model.opt.timestep/control_step
            rm1_vx, rm1_vy= rotate([rm1_vx, rm1_vy], rm1.theta)
            
            rm1_omega = (rm1.theta - rm1.prev_theta)/model.opt.timestep/control_step
            rm1_wheel = mecanum_wheel_velocities(rm1_vx, rm1_vy, rm1_omega)
        else:
            rm1_wheel = np.zeros(4)
        if not rm2_carried:
            rm2_vx = (rm2.x - rm2.prev_x)/model.opt.timestep/control_step
            rm2_vy = (rm2.y - rm2.prev_y)/model.opt.timestep/control_step
            rm2_vx, rm2_vy= rotate([rm2_vx, rm2_vy], rm2.theta)
            rm2_omega = (rm2.theta - rm2.prev_theta)/model.opt.timestep/control_step
            rm2_wheel = mecanum_wheel_velocities(rm2_vx, rm2_vy, rm2_omega)
        else:
            rm2_wheel = np.zeros(4)
        # data.qvel[rm1_wheel_joint_id] =  rm1_wheel+10*(np.random.random(4)-0.5)
        data.qvel[rm1_wheel_joint_id] =  rm1_wheel + 10 * (np.random.random(4) - 0.5)
        data.qvel[rm2_wheel_joint_id] =  rm2_wheel + 10 * (np.random.random(4) - 0.5)

        # print(rm1.theta*180/np.pi, rm1_theta*180/np.pi, ap.theta*180/np.pi)


        
        t += model.opt.timestep*control_step
        
        mujoco.mj_step(model, data, nstep=control_step)
        update([ap, rm1, rm2], [apollo_base_id, rm1_id, rm2_id], data)
        
        if viewer.render_mode=='offscreen':
            viewer.cam.lookat[:] = [ap.x, ap.y, 0.4]
            viewer.cam.distance = 5
            viewer.cam.elevation = -30  # Camera looks straight down  
            viewer.cam.azimuth = int(np.rad2deg(ap.theta))      # Keep it aligned with the world axes
            frames_ap.append(viewer.read_pixels())


            # viewer.cam.lookat[:] = [rm1.x, rm1.y, rm1.z]
            # viewer.cam.distance = 5
            # viewer.cam.elevation = -45  # Camera looks straight down  
            # viewer.cam.azimuth = int(np.rad2deg(rm1.theta))      # Keep it aligned with the world axes
            
            # frames_rm1.append(viewer.read_pixels())


            # viewer.cam.lookat[:] = [rm2.x, rm2.y, rm2.z]
            # viewer.cam.elevation = -45  # Camera looks straight down  
            # viewer.cam.azimuth = int(np.rad2deg(rm2.theta))    # Keep it aligned with the world axes
            # frames_rm2.append(viewer.read_pixels())

        else:
            viewer.cam.lookat[:] = [ap.x, ap.y, 0.4]
            viewer.cam.distance = 5
            viewer.cam.elevation = -30  # Camera looks straight down  
            viewer.cam.azimuth = int(np.rad2deg(ap.theta))
            viewer.render()
    if viewer.render_mode=='offscreen':
        imageio.mimwrite("recordings/apollo_recording.mp4", frames_ap, fps=1/model.opt.timestep/control_step)
        imageio.mimwrite("recordings/rm1_recording.mp4", frames_rm1, fps=1/model.opt.timestep)
        imageio.mimwrite("recordings/rm2_recording.mp4", frames_rm2, fps=1/model.opt.timestep)
    
    viewer.close()
    print('Overall simulation time: ',time.time()-t1)