"""
a grasping task
"""
import taichi as ti
from math import pi
import numpy as np
from difftactile.sensor_model.gripper_kinematics import Gripper
from difftactile.object_model.mpm_plastic import MPMObj
import os
import cv2
import math
import trimesh
import matplotlib.pyplot as plt
from deform_loss import Deform_Loss

Off_screen = False

import argparse
parser = argparse.ArgumentParser()

parser.add_argument("--use_state", action = "store_true", help = "whether to use state loss")
parser.add_argument("--use_tactile", action = "store_true", help = "whether to use tactile loss")
parser.add_argument("--obj_name", default = "J03_2.obj")

args = parser.parse_args()
USE_STATE = args.use_state
USE_TACTILE = args.use_tactile

@ti.data_oriented
class Contact:
    def __init__(self, dt=5e-5, total_steps=300, substeps = 50,  obj=None):
        self.dt = dt
        self.total_steps = total_steps
        self.substeps = substeps
        self.space_scale = 10.0
        self.obj_scale = 4.0
        self.dim = 3
        self.mpm_object = MPMObj(dt=dt, 
                      sub_steps=substeps,
                      obj_name=obj, 
                      space_scale = self.space_scale, 
                      obj_scale = self.obj_scale,
                      density = 1.5,
                      rho = 4.0)
        
        data_path = os.path.join("..", "meshes", "sensors")
        self.gripper = Gripper(data_path, sub_steps=substeps)
        self.deform_loss = Deform_Loss(self.mpm_object)
        

        self.view_phi = 0
        self.view_theta = 0
        self.marker_radius = ti.field(float, shape=())

    
        self.draw_pos = ti.Vector.field(3, dtype=float, shape=(self.mpm_object.n_particles))
        self.kn = ti.field(dtype=float, shape=(), needs_grad=True)
        self.kd = ti.field(dtype=float, shape=(), needs_grad=True)
        self.kt = ti.field(dtype=float, shape=(), needs_grad=True)
        self.friction_coeff = ti.field(dtype=float, shape=(), needs_grad=True)
        self.kn[None] = 34.53
        self.kd[None] = 269.44
        self.kt[None] = 154.78
        self.friction_coeff[None] = 53.85
        self.gripper.fem_sensor1.mu[None] = 1294.01
        self.gripper.fem_sensor1.lam[None] = 9201.11
        self.gripper.fem_sensor2.mu[None] = 1294.01
        self.gripper.fem_sensor2.lam[None] = 9201.11
       
        self.predict_obj_pos = ti.Vector.field(self.dim, float, shape=())
        
        self.contact_idx = ti.Vector.field(2, dtype=int, shape=(self.substeps, self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid))
        self.fem_sensor_pos1 = ti.Vector.field(3, float, shape=(self.gripper.fem_sensor1.n_verts), needs_grad=True)
        self.fem_sensor_pos2 = ti.Vector.field(3, float, shape=(self.gripper.fem_sensor2.n_verts), needs_grad=True)

        # Transloation, Rotation and Width speed of the gripper
        self.p_gripper = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.total_steps), needs_grad=True)
        self.o_gripper = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.total_steps), needs_grad=True)
        self.w_gripper = ti.field(dtype=ti.f32,shape=(self.total_steps), needs_grad=True )

        self.loss = ti.field(float, (), needs_grad=True)
        self.predict_force1 = ti.Vector.field(self.dim, float, shape=(), needs_grad=True)
        self.predict_force2 = ti.Vector.field(self.dim, float, shape=(), needs_grad=True)
        self.contact_force1 = ti.Vector.field(self.dim, float, shape=(), needs_grad=True)
        self.contact_force2 = ti.Vector.field(self.dim, float, shape=(), needs_grad=True)

        self.alpha = ti.field(float, ())
        self.beta = ti.field(float, ())
        self.gamma = ti.field(float, ())

        self.alpha[None] = 5e-4
        self.beta[None] = 5e1
        self.gamma[None] = 1.0
        self.marker_radius[None] = 0.03

        self.norm_eps = 1e-11

        # The indicator flag to detect the contact 
        self.contact_detect_flag = ti.field(float, (), needs_grad=True) 

        # Calculate the marker offset information
        self.marker_offset_norm1 = ti.Vector.field(2, float, shape=(), needs_grad=True) #To get the contact detect flag
        self.marker_offset1 = ti.Vector.field(2, float, shape=(), needs_grad=True)

        self.marker_offset_norm2 = ti.Vector.field(2, float, shape=(), needs_grad=True) #To get the contact detect flag
        self.marker_offset2 = ti.Vector.field(2, float, shape=(), needs_grad=True)


        # The target object pos verifying to lift the object 
        self.target_obj_pos = ti.Vector.field(self.dim, float, shape=())
        self.predict_obj_pos = ti.Vector.field(self.dim, float, shape=(), needs_grad=True)

        #Target grid
        self.target_grid= ti.field(dtype=float, shape=(self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid)) 

    def init(self, path = None):
        rot = [0.0, 0.0, 0.0]
        t = [5.5, 11.3, 5.80]
        # grip = 4.7
        grip = 3.0
        self.gripper.init(rot, t, grip)
        g_trans_h = self.gripper.ee_delta[0].to_numpy()
        g_trans_h[1,3] -= 2.0
        self.gripper.gripper_base.update(g_trans_h)
        g_trans_finger1 = self.gripper.finger1_delta[0].to_numpy()
        g_trans_finger2 = self.gripper.finger2_delta[0].to_numpy()
        self.gripper.gripper_finger1.update(g_trans_finger1)
        self.gripper.gripper_finger2.update(g_trans_finger2)
        self.gripper.init_fem_sensor()

        self.ball_pos = [5.5, 2.3, 5.80]
        self.ball_ori = [-90.0, 0.0, 0.0]
        self.ball_vel = [0.0, 0.0, 0.0]

        self.mpm_object.init(self.ball_pos, self.ball_ori, self.ball_vel)
        y_min = 4.7

        for p in range(self.mpm_object.n_particles):
            y = self.mpm_object.x_0[0, p][1]
            if y< y_min:
                y_min = y
        self.deform_loss.initialize(self.target_grid)

    def generate_target_grid(self):
        self.ball_ori = [-90.0, 0.0, 0.0]
        self.ball_vel = [0.0, 0.0, 0.0]
        self.mpm_object.init(self.target_obj_pos[None], self.ball_ori, self.ball_vel)
        self.mpm_object.compute_grid_m_kernel(0)
        self.target_grid = self.mpm_object.grid_m_s

    def update(self, f):

        self.mpm_object.compute_new_F(f)
        self.mpm_object.svd(f)
        self.mpm_object.p2g(f)
        self.gripper.fem_sensor1.update(f)
        self.gripper.fem_sensor2.update(f)
        self.mpm_object.check_grid_occupy(f)
        self.check_collision(f)
        self.collision(f)
        self.mpm_object.grid_op(f)        
        self.mpm_object.g2p(f)
        self.gripper.fem_sensor1.update2(f)
        self.gripper.fem_sensor2.update2(f)


    def update_grad(self, f):
        self.gripper.fem_sensor1.update2.grad(f)
        self.gripper.fem_sensor2.update2.grad(f)
        self.mpm_object.g2p.grad(f)
        self.mpm_object.grid_op.grad(f)
        self.clamp_grid(f)
        self.collision.grad(f)
        self.gripper.fem_sensor1.update.grad(f)
        self.gripper.fem_sensor2.update.grad(f)
        self.mpm_object.p2g.grad(f)
        self.mpm_object.svd_grad(f)
        self.mpm_object.compute_new_F.grad(f)

    @ti.kernel
    def set_pos_control(self, f:ti.i32):
        self.gripper.d_pos[None] = self.p_gripper[f]
        self.gripper.d_ori[None] = self.o_gripper[f]
        self.gripper.d_gripper[None] = self.w_gripper[f]

    @ti.kernel
    def compute_obj_pos(self, f:ti.i32):
        for p in range(self.mpm_object.n_particles):
            self.predict_obj_pos[None] += (1/self.mpm_object.n_particles)*self.mpm_object.x_0[f, p]
    
    @ti.kernel
    def clamp_grid(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid):
            self.mpm_object.grid_m.grad[f, i, j, k] = ti.math.clamp(self.mpm_object.grid_m.grad[f, i, j, k], -1000.0, 1000.0)

        for i in range(self.gripper.fem_sensor1.n_verts):
            self.gripper.fem_sensor1.pos.grad[f, i] = ti.math.clamp(self.gripper.fem_sensor1.pos.grad[f, i], -1000.0, 1000.0)
            self.gripper.fem_sensor1.vel.grad[f, i] = ti.math.clamp(self.gripper.fem_sensor1.vel.grad[f, i], -1000.0, 1000.0)

        for i in range(self.gripper.fem_sensor2.n_verts):
            self.gripper.fem_sensor2.pos.grad[f, i] = ti.math.clamp(self.gripper.fem_sensor2.pos.grad[f, i], -1000.0, 1000.0)
            self.gripper.fem_sensor2.vel.grad[f, i] = ti.math.clamp(self.gripper.fem_sensor2.vel.grad[f, i], -1000.0, 1000.0)

    @ti.func
    def calculate_contact_force(self, sdf, norm_v, relative_v):
        shear_factor_p0 = ti.Vector([0.0, 0.0, 0.0])
        shear_vel_p0 = ti.Vector([0.0, 0.0, 0.0])
        relative_vel_p0 = relative_v
        normal_vel_p0 = ti.max(norm_v.dot(relative_vel_p0), 0)
        normal_factor_p0 = -(self.kn[None] + self.kd[None] * normal_vel_p0)* sdf * norm_v
        shear_vel_p0 = relative_vel_p0 - norm_v.dot(relative_vel_p0) * norm_v
        shear_vel_norm_p0 = shear_vel_p0.norm(self.norm_eps)
        if shear_vel_norm_p0 > 1e-4:
            shear_factor_p0 = 1.0*(shear_vel_p0/shear_vel_norm_p0) * ti.min(self.kt[None] * shear_vel_norm_p0, self.friction_coeff[None]*normal_factor_p0.norm())
        ext_v = normal_factor_p0 + shear_factor_p0
        return ext_v, normal_factor_p0, shear_factor_p0

    @ti.kernel
    def compute_contact_force(self, f:ti.i32):
        for i in range(self.gripper.fem_sensor1.num_triangles):
            a, b, c = self.gripper.fem_sensor1.contact_seg[i]
            self.contact_force1[None] += 1/6 * self.gripper.fem_sensor1.external_force_field[f,a]
            self.contact_force1[None] += 1/6 * self.gripper.fem_sensor1.external_force_field[f,b]
            self.contact_force1[None] += 1/6 * self.gripper.fem_sensor1.external_force_field[f,c] 
        for i in range(self.gripper.fem_sensor2.num_triangles):
            a, b, c = self.gripper.fem_sensor2.contact_seg[i]
            self.contact_force2[None] += 1/6 * self.gripper.fem_sensor2.external_force_field[f,a] 
            self.contact_force2[None] += 1/6 * self.gripper.fem_sensor2.external_force_field[f,b] 
            self.contact_force2[None] += 1/6 * self.gripper.fem_sensor2.external_force_field[f,c] 
    
    @ti.kernel
    def compute_force_loss(self):
        self.predict_force1[None] = self.gripper.fem_sensor1.inv_rot[None] @ self.contact_force1[None]
        self.predict_force2[None] = self.gripper.fem_sensor2.inv_rot[None] @ self.contact_force2[None]
        if abs(self.predict_force1[None][0]) > 1e-3 and abs(self.predict_force2[None][0]) >1e-3:
            self.loss[None] += self.beta[None]* ( abs(self.predict_force1[None][0])/abs(self.friction_coeff[None]*self.predict_force1[None][1]) + abs(self.predict_force2[None][0])/abs(self.friction_coeff[None]*self.predict_force2[None][1]))
        else:
            self.loss[None] += 0

    @ti.kernel
    def compute_marker_motion(self):
        for p in range(self.gripper.fem_sensor1.num_markers):
            self.marker_offset1[None] += 1/self.gripper.fem_sensor1.num_markers*(self.gripper.fem_sensor1.predict_markers[p] - self.gripper.fem_sensor1.virtual_markers[p])
            self.marker_offset2[None] += 1/self.gripper.fem_sensor2.num_markers*(self.gripper.fem_sensor2.predict_markers[p] - self.gripper.fem_sensor2.virtual_markers[p])
            self.marker_offset_norm1[None] += 1/self.gripper.fem_sensor1.num_markers*abs(self.gripper.fem_sensor1.predict_markers[p] - self.gripper.fem_sensor1.virtual_markers[p])
            self.marker_offset_norm2[None] += 1/self.gripper.fem_sensor2.num_markers*abs(self.gripper.fem_sensor2.predict_markers[p] - self.gripper.fem_sensor2.virtual_markers[p])


    @ti.kernel
    def compute_deform_loss(self):
        self.loss[None] += self.alpha[None]*self.deform_loss.loss[None]
        

    @ti.kernel
    def compute_marker_loss(self): 
        self.loss[None] += self.beta[None]*((self.marker_offset1[None][0])**2 + (self.marker_offset1[None][1])**2 + (self.marker_offset2[None][0])**2 + (self.marker_offset2[None][1])**2)

    @ti.kernel
    def compute_obj_pos(self, f:ti.i32):
        for p in range(self.mpm_object.n_particles):
            self.predict_obj_pos[None] += (1/self.mpm_object.n_particles)*self.mpm_object.x_0[f, p]

    @ti.kernel
    def compute_pos_loss(self):
        self.loss[None] += self.gamma[None]*( (self.predict_obj_pos[None][0] - self.target_obj_pos[None][0])**2 + (self.predict_obj_pos[None][1] - self.target_obj_pos[None][1])**2 + (self.predict_obj_pos[None][2] - self.target_obj_pos[None][2])**2)


    
    @ti.kernel
    def check_collision(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid):
            if self.mpm_object.grid_occupy[f, i, j, k] == 1:
                cur_p = ti.Vector([(i+0.5)*self.mpm_object.dx_0, (j+0.5)*self.mpm_object.dx_0, (k+0.5)*self.mpm_object.dx_0])
                min_idx1 = self.gripper.fem_sensor1.find_closest(cur_p, f)
                min_idx2 = self.gripper.fem_sensor2.find_closest(cur_p, f)
                self.contact_idx[f, i, j, k] = [min_idx1, min_idx2]
    
    
    @ti.kernel
    def collision(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid):
            if self.mpm_object.grid_occupy[f, i, j, k] == 1:
                cur_p = ti.Vector([(i+0.5)*self.mpm_object.dx_0, (j+0.5)*self.mpm_object.dx_0, (k+0.5)*self.mpm_object.dx_0])

                cur_v = self.mpm_object.grid_v_in[f, i, j, k] / (self.mpm_object.grid_m[f, i, j, k]+self.mpm_object.eps)
                min_idx1, min_idx2 = self.contact_idx[f, i, j, k]
                cur_sdf2, cur_norm_v2, cur_relative_v2, contact_flag2 = self.gripper.fem_sensor2.find_sdf(cur_p, cur_v, min_idx2, f)
                cur_sdf1, cur_norm_v1, cur_relative_v1, contact_flag1 = self.gripper.fem_sensor1.find_sdf(cur_p, cur_v, min_idx1, f)
                if contact_flag1:
                    ext_v1, ext_n1, ext_t1 = self.calculate_contact_force(cur_sdf1, -1*cur_norm_v1, -1*cur_relative_v1)
                    self.mpm_object.update_contact_force(ext_v1, f, i, j, k)
                    self.gripper.fem_sensor1.update_contact_force(min_idx1, -1*ext_v1, f)
                    
    
                    
                if contact_flag2:
                    ext_v2, ext_n2, ext_t2 = self.calculate_contact_force(cur_sdf2, -1*cur_norm_v2, -1*cur_relative_v2)
                    self.mpm_object.update_contact_force(ext_v2, f, i, j, k)
                    self.gripper.fem_sensor2.update_contact_force(min_idx2, -1*ext_v2, f)

    @ti.kernel
    def add_virtual_force(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid):
            if self.mpm_object.grid_occupy[f, i, j, k] == 1:
                self.mpm_object.grid_f[f, i, j, k] = ti.Vector([0.0, 0.0, 0.0]) * self.dt


    @ti.kernel
    def draw_particles(self, f:ti.i32):
        for p in range(self.mpm_object.n_particles):
            self.draw_pos[p] = self.mpm_object.x_0[f, p]
              
    def draw_markers(self, init_markers, cur_markers, gui):
        img_height = 480
        img_width = 640
        scale = img_width
        rescale = 1.8
        draw_points = rescale * (init_markers - [320, 240]) / scale + [0.5, 0.5]
        offset = rescale * (cur_markers - init_markers) / scale
        gui.circles(draw_points, radius=2, color=0xf542a1)
        gui.arrows(draw_points, 10.0*offset, radius=2, color=0xe6c949)
        
    def draw_markers_cv(self, tracked_markers, init_markers, showScale):
        img = np.zeros((480, 640, 3))
        markerCenter = np.around(init_markers[:, 0:2]).astype(np.int16)
        for i in range(init_markers.shape[0]):
            marker_motion = tracked_markers[i] - init_markers[i]
            cv2.arrowedLine(img, (markerCenter[i,0], markerCenter[i,1]), \
                (int(init_markers[i,0]+marker_motion[0]*showScale), int(init_markers[i,1]+marker_motion[1]*showScale)),\
                    (0, 255, 255), 2)
        return img
    
    def load_target(self):
        self.target_obj_pos[None] = ti.Vector([5.5, 4.5, 5.8]) # only use normal force

    
             
    @ti.kernel
    def clear_loss_grad(self):
        self.kn.grad[None] = 0.0
        self.kd.grad[None] = 0.0
        self.kt.grad[None] = 0.0
        self.friction_coeff.grad[None] = 0.0

        self.loss[None] = 0.0
        self.loss.grad[None] = 1.0

        self.contact_force1.grad[None].fill(0.0)
        self.predict_force1.grad[None].fill(0.0)
        self.contact_force2.grad[None].fill(0.0)
        self.predict_force2.grad[None].fill(0.0)

        self.marker_offset1.grad[None].fill(0.0)
        self.marker_offset2.grad[None].fill(0.0)
        self.marker_offset_norm1.grad[None].fill(0.0)
        self.marker_offset_norm2.grad[None].fill(0.0)

        self.p_gripper.grad.fill(0.0)
        self.o_gripper.grad.fill(0.0)
        self.w_gripper.grad.fill(0.0)

    def clear_traj_grad(self):
        self.mpm_object.clear_loss_grad()
        self.gripper.clear_loss_grad()
        self.gripper.fem_sensor1.clear_loss_grad()
        self.gripper.fem_sensor2.clear_loss_grad()
        self.clear_loss_grad()
        self.deform_loss.clear_loss_grad()
    
    def clear_all_grad(self):
        self.clear_traj_grad()
        self.gripper.clear_step_grad(self.substeps)
        self.gripper.fem_sensor1.clear_step_grad(self.substeps)
        self.gripper.fem_sensor2.clear_step_grad(self.substeps)
        self.mpm_object.clear_step_grad(self.substeps)


    def clear_step_grad(self):
        self.clear_traj_grad()
        self.gripper.clear_step_grad(self.substeps-1)
        self.gripper.fem_sensor1.clear_step_grad(self.substeps-1)
        self.gripper.fem_sensor2.clear_step_grad(self.substeps-1)
        self.mpm_object.clear_step_grad(self.substeps-1)

        
    def reset(self):
        self.mpm_object.reset()
        self.gripper.fem_sensor1.reset_contact()
        self.gripper.fem_sensor2.reset_contact()
        self.contact_idx.fill(-1)
        self.predict_obj_pos[None].fill(0.0)
        self.contact_detect_flag[None] = 0.0

        self.contact_force1[None].fill(0.0)
        self.predict_force1[None].fill(0.0)
        self.contact_force2[None].fill(0.0)
        self.predict_force2[None].fill(0.0)

        self.marker_offset1[None].fill(0.0)
        self.marker_offset2[None].fill(0.0)
        self.marker_offset_norm1[None].fill(0.0)
        self.marker_offset_norm2[None].fill(0.0)
        
    def memory_to_cache(self, t):
        self.gripper.memory_to_cache(t)
        self.gripper.fem_sensor1.memory_to_cache(t)
        self.gripper.fem_sensor2.memory_to_cache(t)
        self.mpm_object.memory_to_cache(t)
        
    def memory_from_cache(self, t):
        self.gripper.memory_from_cache(t)
        self.gripper.fem_sensor1.memory_from_cache(t)
        self.gripper.fem_sensor2.memory_from_cache(t)
        self.mpm_object.memory_from_cache(t)
    
    @ti.kernel
    def get_gripper_pos(self, f:ti.i32):
        for item in range(self.gripper.fem_sensor1.n_verts):
            self.fem_sensor_pos1[item] = self.gripper.fem_sensor1.pos[f, item]
        for item in range(self.gripper.fem_sensor2.n_verts):
            self.fem_sensor_pos2[item] = self.gripper.fem_sensor2.pos[f, item]
            
    @ti.kernel
    def init_control_parameters(self):
        # half_traj = self.total_steps //5
        vx = 0.0; vy = 0.0; vz = 0.0
        rx = 0.0; ry = 0.0; rz = 0.0
        ws = 10   # ws > 0, gripper width will get shorter
        
        for i in range(0, self.total_steps):
            self.p_gripper[i] = ti.Vector([vx, vy, vz])
            self.o_gripper[i] = ti.Vector([rx, ry, rz])
            self.w_gripper[i] = ws


    @ti.kernel
    def reset_initial_control(self, t:ti.i32):

        vx = 0.0; vy = 5.0; vz = 0.0
        rx = 0.0; ry = 0.0; rz = 0.0
        ws = 0.0   # ws > 0, gripper width will get shorter
        
        for i in range(t+1, self.total_steps):
            self.p_gripper[i] = ti.Vector([vx, vy, vz])
            self.o_gripper[i] = ti.Vector([rx, ry, rz])
            self.w_gripper[i] = ws


    def update_visualization(self, f:ti.i32):
        g_trans_h = self.gripper.ee_delta[f].to_numpy()
        self.gripper.gripper_base.update(g_trans_h)
        g_trans_finger1 = self.gripper.finger1_delta[f].to_numpy()
        g_trans_finger2 = self.gripper.finger2_delta[f].to_numpy()
        self.gripper.gripper_finger1.update(g_trans_finger1)
        self.gripper.gripper_finger2.update(g_trans_finger2)




def hasnan(vector):
    for i in range(3):
        if math.isnan(vector[i]):
            return True
        else:
            return False



ti.init(arch=ti.gpu, device_memory_GB=4)


def main():
    if not Off_screen:
    
        window = ti.ui.Window("Grasping",(1024,1024))
        canvas = window.get_canvas()
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()
        camera.position(22.5, 25.5, 30.0)
        camera.lookat(2.5, 2.5, 5.0)
        camera.projection_mode(ti.ui.ProjectionMode.Perspective)

        gui_contact = ti.GUI("Force Map")
    obj_name = args.obj_name
    sub_steps = 50
    total_steps = 600
    opt_steps = 200
    dt = 5e-5
    contact_model = Contact(total_steps=total_steps, substeps= sub_steps, dt=dt, obj=obj_name)
    reset = False
    cnt = 0
    
   
    
    contact_model.init_control_parameters()
    contact_model.load_target()
    contact_model.generate_target_grid()


    form_loss = 0
    losses = []


    
    for opts in range(opt_steps):
        contact_model.generate_target_grid()
        
        contact_model.init()
        contact_model.clear_all_grad()
        reset_flag = False
        
        
        
        for ts in range(total_steps-1):
          
            print("FP", ts)
            contact_model.set_pos_control(ts)
            for ss in range(sub_steps-1):
                contact_model.gripper.kinematic(ss)
                contact_model.update_visualization(ss+1)
            contact_model.gripper.set_sensor_vel(0)
            contact_model.gripper.set_sensor_pos(sub_steps-1)
            contact_model.reset()
            for ss in range(sub_steps-1):
                contact_model.update(ss)
            contact_model.memory_to_cache(ts)


            contact_model.predict_obj_pos[None].fill(0.0)
            contact_model.compute_obj_pos(sub_steps - 1)
            print("predict_obj_pos", contact_model.predict_obj_pos[None])


            form_loss = contact_model.loss[None]
            contact_model.compute_contact_force(sub_steps - 2)
            form_loss = contact_model.loss[None]
            contact_model.compute_force_loss()
            print("Force loss", contact_model.loss[None]-form_loss)
            print("Predict force", contact_model.predict_force1[None], contact_model.predict_force2[None])
            f_shear = ti.Vector([0.0, 0.0])
            f_shear[0] = contact_model.predict_force1[None][0]
            f_shear[1] = contact_model.predict_force1[None][2]
            print("shear force", f_shear.norm(), "norm force", contact_model.predict_force1[None][1], "mu multiple norm force", contact_model.friction_coeff[None]*contact_model.predict_force1[None][1])
       

            contact_model.clear_all_grad()
            form_loss = contact_model.loss[None]
            contact_model.deform_loss.compute_loss_kernel(sub_steps - 1)
            contact_model.compute_deform_loss()

            print("deform loss", contact_model.loss[None]-form_loss)
            THRE = 1e3

            if opts == 0 and  abs(contact_model.predict_force1[None][0]) > THRE and abs(contact_model.predict_force2[None][0]) >THRE  and reset_flag== False:   #1e6 is nice
                reset_flag = True
                contact_model.reset_initial_control(ts)


 
            if not Off_screen:
                init_2d = contact_model.gripper.fem_sensor1.virtual_markers.to_numpy()
                marker_2d = contact_model.gripper.fem_sensor1.predict_markers.to_numpy()
                contact_model.draw_markers(init_2d, marker_2d, gui_contact)
                gui_contact.show()
                camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.RMB)
                scene.set_camera(camera)
                scene.ambient_light((0.8, 0.8, 0.8))
                scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
                contact_model.draw_particles(0)
                contact_model.get_gripper_pos(contact_model.substeps-1)
                scene.particles(contact_model.draw_pos, color = (0.68, 0.2==6, 0.19), radius = 0.02)
                scene.mesh(contact_model.gripper.gripper_base.ti_vertices, contact_model.gripper.gripper_base.ti_faces, contact_model.gripper.gripper_base.ti_normals)
                scene.mesh(contact_model.gripper.gripper_finger1.ti_vertices, contact_model.gripper.gripper_finger1.ti_faces, contact_model.gripper.gripper_finger1.ti_normals)
                scene.mesh(contact_model.gripper.gripper_finger2.ti_vertices, contact_model.gripper.gripper_finger2.ti_faces, contact_model.gripper.gripper_finger2.ti_normals)
                scene.particles(contact_model.fem_sensor_pos1, color = (0.1, 0.5, 0.5), radius = 0.05)
                scene.particles(contact_model.fem_sensor_pos2, color = (0.1, 0.5, 0.5), radius = 0.05)
                canvas.scene(scene)
                window.show()

        print("Begin backward")
        loss_frame = 0

        if USE_STATE:
            contact_model.clear_all_grad()
            form_loss = contact_model.loss[None]
            contact_model.deform_loss.compute_loss_kernel(sub_steps - 1)
            contact_model.compute_deform_loss()
            print("loss", contact_model.loss[None] - form_loss)
            contact_model.compute_deform_loss.grad()
            contact_model.deform_loss.compute_loss_kernel_grad(sub_steps -1)
            loss_frame += contact_model.loss[None]


        for ts in range(total_steps-2, -1, -1):
                print("BP", ts)
                contact_model.clear_step_grad()

                if USE_TACTILE:
                    contact_model.compute_contact_force(sub_steps - 2)
                    contact_model.compute_force_loss()

                if USE_TACTILE:
                    contact_model.compute_force_loss.grad()
                    contact_model.compute_contact_force.grad(sub_steps - 2)
               
                # Loss grad
                for ss in range(sub_steps-2, -1, -1):
                    contact_model.update_grad(ss)

                contact_model.gripper.set_sensor_vel_grad(0)
                for ss in range(sub_steps-2, -1, -1):
                    contact_model.gripper.kinematic_grad(ss)
            
                contact_model.gripper.set_sensor_pos_grad(sub_steps-1)
                
                contact_model.set_pos_control.grad(ts)

                grad_p = contact_model.p_gripper.grad[ts]
                grad_o = contact_model.o_gripper.grad[ts]
                grad_w = contact_model.w_gripper.grad[ts]

                lr_p =1e1
                lr_o = 1e-5
                lr_w = 1e1

                loss_frame += contact_model.loss[None]
                print("# BP Iter: ", ts, " loss: ", contact_model.loss[None])
                print("P/O grads: ", grad_p, grad_o, grad_w)

                if not hasnan(grad_p):
                    contact_model.p_gripper[ts] -= lr_p * grad_p
                if not hasnan(grad_o):
                    contact_model.o_gripper[ts] -= lr_o * grad_o
                if not math.isnan(grad_w):
                    contact_model.w_gripper[ts] -= lr_w * grad_w

                print("P/O updated: ", contact_model.p_gripper[ts], contact_model.o_gripper[ts], contact_model.w_gripper[ts])

                if (ts -1) >= 0:
                    contact_model.memory_from_cache(ts -1)
                    contact_model.set_pos_control(ts-1)
                    for ss in range(sub_steps-1):
                        contact_model.gripper.kinematic(ss)
                    contact_model.gripper.set_sensor_vel(0)
                    contact_model.gripper.set_sensor_pos_bp()
                    contact_model.reset()
                    for ss in range(sub_steps-1):
                        contact_model.update(ss)
            
        
        losses.append(loss_frame)
        print("# Iter ", opts, "Opt step loss: ", loss_frame)


        if not os.path.exists(f"grasp_plastic_{args.use_state}_tactile_{args.use_tactile}_{args.obj_name}"):
                os.mkdir(f"grasp_plastic_{args.use_state}_tactile_{args.use_tactile}_{args.obj_name}")

        if not os.path.exists(f"results"):
            os.mkdir(f"results")
        
        if opts %5 == 0 or opts == opt_steps-1:
            plt.title("Trajectory Optimization")
            plt.ylabel("Loss")
            plt.xlabel("Iter") # "Gradient Descent Iterations"
            plt.plot(losses)
            plt.savefig(os.path.join(f"grasp_plastic_{args.use_state}_tactile_{args.use_tactile}_{args.obj_name}",f"grasp_plastic_{args.use_state}_tactile_{args.use_tactile}_{opts}.png"))
            np.save(os.path.join(f"grasp_plastic_{args.use_state}_tactile_{args.use_tactile}_{args.obj_name}", f"control_p_gripper_{opts}.npy"), contact_model.p_gripper.to_numpy())
            np.save(os.path.join(f"grasp_plastic_{args.use_state}_tactile_{args.use_tactile}_{args.obj_name}", f"control_o_gripper_{opts}.npy"), contact_model.o_gripper.to_numpy())
            np.save(os.path.join(f"grasp_plastic_{args.use_state}_tactile_{args.use_tactile}_{args.obj_name}", f"control_w_gripper_{opts}.npy"), contact_model.w_gripper.to_numpy())
            np.save(os.path.join(f"grasp_plastic_{args.use_state}_tactile_{args.use_tactile}_{args.obj_name}", f"losses_{opts}.npy"), np.array(losses))

        if loss_frame <= np.min(losses):
            best_p = contact_model.p_gripper.to_numpy()
            best_o = contact_model.o_gripper.to_numpy()
            best_w = contact_model.w_gripper.to_numpy()
            np.save(os.path.join(f"grasp_plastic_{args.use_state}_tactile_{args.use_tactile}_{args.obj_name}","control_p_gripper_best.npy"), best_p)
            np.save(os.path.join(f"grasp_plastic_{args.use_state}_tactile_{args.use_tactile}_{args.obj_name}","control_o_gripper_best.npy"), best_o)
            np.save(os.path.join(f"grasp_plastic_{args.use_state}_tactile_{args.use_tactile}_{args.obj_name}","control_w_gripper_best.npy"), best_w)
            print("Best traj saved!")

if __name__ == "__main__":
    main()