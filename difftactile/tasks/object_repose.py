"""
an object reposing task
"""
import taichi as ti
from math import pi
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os

off_screen = False
# if off_screen:    
#     os.environ["PYOPENGL_PLATFORM"] = "egl"

import trimesh

from difftactile.sensor_model.fem_sensor import FEMDomeSensor
from difftactile.object_model.rigid_dynamic import RigidObj

import argparse

TI_TYPE = ti.f32
NP_TYPE = np.float32



@ti.data_oriented
class Contact:
    def __init__(self, use_state, use_tactile, dt=5e-5, total_steps=300, sub_steps=50,obj=None):
        self.iter = 0
        
        self.dt = dt
        self.total_steps = total_steps
        self.prepare_step = 300
        self.sub_steps = sub_steps
        self.dim = 3
        self.fem_sensor1 = FEMDomeSensor(dt, sub_steps)
        self.space_scale = 10.0
        self.obj_scale = 4.0
        self.use_state = use_state
        self.use_tactile = use_tactile

        self.mpm_object = RigidObj(dt=dt, 
                      sub_steps=sub_steps,
                      obj_name=obj, 
                      space_scale = self.space_scale, 
                      obj_scale = self.obj_scale,
                      density = 1.50,
                      rho = 0.3)
        
        self.alpha = ti.field(float, ())
        self.beta = ti.field(float, ())
        self.alpha[None] = 1e1
        self.beta[None] = 5e-12
        self.num_sensor = 1
        self.init()

        self.view_phi = 0
        self.view_theta = 0
        self.kn = ti.field(dtype=float, shape=(), needs_grad=True)
        self.kd = ti.field(dtype=float, shape=(), needs_grad=True)
        self.kt = ti.field(dtype=float, shape=(), needs_grad=True)
        self.friction_coeff = ti.field(dtype=float, shape=(), needs_grad=True)#0.5

        self.kn[None] = 55.0
        self.kd[None] = 269.44
        self.kt[None] = 108.72
        self.friction_coeff[None] = 14.16
        E = 0.8e4
        nu = 0.4
        self.fem_sensor1.set_material_params(E, nu)

        self.contact_idx = ti.Vector.field(self.num_sensor, dtype=int, shape=(self.sub_steps, self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid)) # indicating the contact seg idx of fem sensor for closested triangle mesh
        self.total_ext_f = ti.Vector.field(3, dtype=float, shape=())

        # control parameters
        self.dim = 3
        ### position control
        self.p_sensor1 = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.total_steps), needs_grad=True)
        self.o_sensor1 = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.total_steps), needs_grad=True)

        # for grad backward
        self.loss = ti.field(float, (), needs_grad=True)
        self.angle = ti.field(float, (self.total_steps), needs_grad=True)

        # The indicator flag to detect the contact 
        self.contact_detect_flag = ti.field(float, (), needs_grad=True) 

      

        # Calculate the contact force, remember that the force need to calibrate in the sensor coordinate
        self.predict_force1 = ti.Vector.field(self.dim, float, (self.total_steps), needs_grad=True)
        self.contact_force1 = ti.Vector.field(self.dim, float, (), needs_grad=True)
       

        # visualization
        self.surf_offset1 = ti.Vector.field(2, float, self.fem_sensor1.num_triangles)
        self.surf_init_pos1 = ti.Vector.field(2, float, self.fem_sensor1.num_triangles)
        self.press_offset1 = ti.field(float, self.fem_sensor1.num_triangles)

        self.draw_pos2 = ti.Vector.field(2, float, self.fem_sensor1.n_verts) # elastomer1's pos
        self.draw_pos3 = ti.Vector.field(2, float, self.mpm_object.n_particles) # object's particle

        # 3d viz 
        self.draw_pos_3d = ti.Vector.field(3, dtype=float, shape=(self.mpm_object.n_particles))
        self.draw_fem1_3d = ti.Vector.field(3, dtype=float, shape=(self.fem_sensor1.n_verts))

        self.contact_grid = ti.field(dtype=int, shape=(self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid))
        self.draw_grid_3d = ti.Vector.field(3, dtype=float, shape=(self.mpm_object.n_grid**3))
        self.color_grid_3d = ti.Vector.field(3, dtype=float, shape=(self.mpm_object.n_grid**3))
        self.color_fem1_3d = ti.Vector.field(3, dtype=float, shape=(self.fem_sensor1.n_verts))

        

        self.norm_eps = 1e-11  #Help the convergence of normalization

        self.target_force1 = ti.Vector.field(self.dim, float, shape=(self.total_steps))
        self.target_angle = ti.field(float, ())

        self.wf = 1e-6

    

        self.angle_x = ti.field(float, ())
        self.angle_y = ti.field(float, ())
        self.angle_z = ti.field(float, ())     
            

    def init(self):
        self.ball_pos = [3.2, 1.0, 5.0]
        self.ball_ori = [0.0, 0.0, 90.0]
        self.ball_vel = [0.0, 0.0, 0.0]
        self.mpm_object.init(self.ball_pos, self.ball_ori, self.ball_vel)

        ### extract the height of the obj
        obj_pos = self.mpm_object.x_0.to_numpy()[0,:]
        sensor_pos = self.fem_sensor1.init_x.to_numpy()

        self.obj_x = np.max(obj_pos[:,0]) - np.min(obj_pos[:,0])
        self.obj_y = np.max(obj_pos[:,1]) - np.min(obj_pos[:,1])
        self.obj_z = np.max(obj_pos[:,2]) - np.min(obj_pos[:,2])


        rx1 = 0.0
        ry1 = 0.0
        rz1 = 90.0
        t_dx1 = 7.0#7.95
        t_dy1 = 1.5
        t_dz1 = 5.0

        self.fem_sensor1.init(rx1, ry1, rz1, t_dx1, t_dy1, t_dz1)


        self.in_contact = False
        self.contact_timestamp = 0

    def load_pos_control(self):


        cur_pos_1 = np.load("control_pos_best.npy")
        cur_ori_1 = np.load("control_ori_best.npy")
        for i in range(len(cur_pos_1)):
            if np.isnan(cur_pos_1[i]).any() or np.isnan(cur_ori_1[i]).any():
                cur_pos_1[i] = cur_pos_1[i-1]
                cur_ori_1[i] = cur_ori_1[i-1]
        self.p_sensor1.from_numpy(cur_pos_1)
        self.o_sensor1.from_numpy(cur_ori_1)

    @ti.kernel
    def init_pos_control(self):

        # initial position for sensor 1 & 2

        #Pressing 
        vx1 = 0.0; vy1 = 1.5; vz1 = 0.0

        rx1 = 0.0; ry1 = 0.0; rz1 = 0.0

        for i in range(0, self.prepare_step):

            self.p_sensor1[i] = ti.Vector([vx1, vy1, vz1])
            self.o_sensor1[i] = ti.Vector([rx1, ry1, rz1])

        vx1 = 1.0; vy1 = 0.5; vz1 = 0.0

        rx1 = 0.0; ry1 = 0.0; rz1 = 0.0

        for i in range(self.prepare_step, self.total_steps):
            self.p_sensor1[i] = ti.Vector([vx1, vy1, vz1])
            self.o_sensor1[i] = ti.Vector([rx1, ry1, rz1])

        


        
    @ti.kernel
    def set_pos_control(self, f:ti.i32):
        self.fem_sensor1.d_pos[None] = self.p_sensor1[f]
        self.fem_sensor1.d_ori[None] = self.o_sensor1[f]
        
        
    def update(self, f):
        self.mpm_object.compute_new_F(f)
        self.mpm_object.svd(f)
        self.mpm_object.p2g(f)
        self.fem_sensor1.update(f)
        # calculate the collision based on current updated status, and then update external forces for next time step
        self.mpm_object.check_grid_occupy(f)
        self.check_collision(f)
        self.collision(f)
        self.mpm_object.grid_op(f)
        self.mpm_object.g2p(f)
        self.mpm_object.compute_COM(f)
        self.mpm_object.compute_H(f)
        self.mpm_object.compute_H_svd(f)
        self.mpm_object.compute_R(f)
        self.fem_sensor1.update2(f)


    def update_grad(self, f):
        self.fem_sensor1.update2.grad(f)
        self.mpm_object.compute_R.grad(f)
        self.mpm_object.compute_H_svd_grad(f)
        self.mpm_object.compute_H.grad(f)
        self.mpm_object.compute_COM.grad(f)
        self.mpm_object.g2p.grad(f)
        self.mpm_object.grid_op.grad(f)
        self.clamp_grid(f)
        self.collision.grad(f)
        self.fem_sensor1.update.grad(f)
        self.mpm_object.p2g.grad(f)
        self.mpm_object.svd_grad(f)
        self.mpm_object.compute_new_F.grad(f)

    @ti.kernel
    def clamp_grid(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid):
            self.mpm_object.grid_m.grad[f, i, j, k] = ti.math.clamp(self.mpm_object.grid_m.grad[f, i, j, k], -1000.0, 1000.0)
        for i in range(self.fem_sensor1.n_verts):
            self.fem_sensor1.pos.grad[f, i] = ti.math.clamp(self.fem_sensor1.pos.grad[f, i], -1000.0, 1000.0)
            self.fem_sensor1.vel.grad[f, i] = ti.math.clamp(self.fem_sensor1.vel.grad[f, i], -1000.0, 1000.0)


    @ti.kernel
    def clear_state_loss_grad(self):
        self.angle.fill (0.0)
        self.angle.grad.fill( 0.0)
        
  
    @ti.kernel
    def clear_loss_grad(self):
        self.kn.grad[None] = 0.0
        self.kd.grad[None] = 0.0
        self.kt.grad[None] = 0.0
        self.friction_coeff.grad[None] = 0.0
        self.contact_detect_flag.grad[None] = 0.0
        self.contact_force1.grad[None].fill(0.0)
        self.predict_force1.grad.fill(0.0)

        self.loss[None] = 0.0
        self.loss.grad[None] = 1.0
        self.p_sensor1.grad.fill(0.0)
        self.o_sensor1.grad.fill(0.0)

        self.angle_x[None] = 0.0
        self.angle_y[None] = 0.0
        self.angle_z[None] = 0.0
 

    def clear_traj_grad(self):
        self.fem_sensor1.clear_loss_grad()
        self.mpm_object.clear_loss_grad()
        self.clear_loss_grad()


    def clear_all_grad(self):
        self.clear_traj_grad()
        self.fem_sensor1.clear_step_grad(self.sub_steps)
        self.mpm_object.clear_step_grad(self.sub_steps)

    def reset(self):
        self.fem_sensor1.reset_contact()
        self.mpm_object.reset()
        self.contact_idx.fill(-1)
        self.contact_detect_flag[None] = 0.0

        self.contact_force1[None].fill(0.0)
        self.predict_force1.fill(0.0)

    


    @ti.kernel
    def compute_contact_force(self, f:ti.i32):
        for i in range(self.fem_sensor1.num_triangles):
            a, b, c = self.fem_sensor1.contact_seg[i]
            self.contact_force1[None] += 1/6 * self.fem_sensor1.external_force_field[f,a] 
            self.contact_force1[None] += 1/6 * self.fem_sensor1.external_force_field[f,b] 
            self.contact_force1[None] += 1/6 * self.fem_sensor1.external_force_field[f,c] 



    @ti.kernel
    def compute_force_loss(self, t:ti.i32):
        self.predict_force1[t] = self.fem_sensor1.inv_rot[None] @ self.contact_force1[None]
        
        self.loss[None] += self.beta[None] *((self.predict_force1[t][1] - self.target_force1[t][1])**2 + (self.predict_force1[t][0] - self.target_force1[t][0])**2)


    def load_target(self):
        
        for ts in range(0, self.prepare_step):
            self.target_force1[ts] = ti.Vector([0.0, -30000.0, 0.0])
           

        for ts in range(self.prepare_step, self.total_steps):
            self.target_force1[ts] = ti.Vector([-25000.0, -1000.0, 0.0])
        self.target_angle[None] = 1.57

    @ti.func
    def calculate_contact_force(self, sdf, norm_v, relative_v):
        shear_factor_p0 = ti.Vector([0.0, 0.0, 0.0])
        shear_vel_p0 = ti.Vector([0.0, 0.0, 0.0])

        relative_vel_p0 = relative_v
        normal_vel_p0 = ti.max(norm_v.dot(relative_vel_p0), 0)

        normal_factor_p0 = -(self.kn[None] + self.kd[None] * normal_vel_p0)* sdf * norm_v

        ### also add spring
        shear_vel_p0 = relative_vel_p0 - norm_v.dot(relative_vel_p0) * norm_v
        shear_vel_norm_p0 = shear_vel_p0.norm(self.norm_eps)
        if shear_vel_norm_p0 > 1e-4:
            shear_factor_p0 = 1.0*(shear_vel_p0/shear_vel_norm_p0) * ti.min(self.kt[None] * shear_vel_norm_p0, self.friction_coeff[None]*normal_factor_p0.norm(self.norm_eps))
        ext_v = normal_factor_p0 + shear_factor_p0
        return ext_v, normal_factor_p0, shear_factor_p0

    @ti.kernel
    def check_collision(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid):
            if self.mpm_object.grid_occupy[f, i, j, k] == 1:
                cur_p = ti.Vector([(i+0.5)*self.mpm_object.dx_0, (j+0.5)*self.mpm_object.dx_0, (k+0.5)*self.mpm_object.dx_0])
                min_idx1 = self.fem_sensor1.find_closest(cur_p, f)
                self.contact_idx[f, i, j, k] = min_idx1


    @ti.kernel
    def collision(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.mpm_object.n_grid, self.mpm_object.n_grid, self.mpm_object.n_grid):
            if self.mpm_object.grid_occupy[f, i, j, k] == 1:
                cur_p = ti.Vector([(i+0.5)*self.mpm_object.dx_0, (j+0.5)*self.mpm_object.dx_0, (k+0.5)*self.mpm_object.dx_0])
                cur_v = self.mpm_object.grid_v_in[f, i, j, k] / (self.mpm_object.grid_m[f, i, j, k]+self.mpm_object.eps)
                min_idx1 = self.contact_idx[f, i, j, k]
                cur_sdf1, cur_norm_v1, cur_relative_v1, contact_flag1 = self.fem_sensor1.find_sdf(cur_p, cur_v, min_idx1, f)
                if contact_flag1:
                    ext_v1, ext_n1, ext_t1 = self.calculate_contact_force(cur_sdf1, -1*cur_norm_v1, -1*cur_relative_v1)
                    self.mpm_object.update_contact_force(ext_v1, f, i, j, k)
                    self.fem_sensor1.update_contact_force(min_idx1, -1*ext_v1, f)



    def memory_to_cache(self, t):
        self.fem_sensor1.memory_to_cache(t)
        self.mpm_object.memory_to_cache(t)

    def memory_from_cache(self, t):
        self.fem_sensor1.memory_from_cache(t)
        self.mpm_object.memory_from_cache(t)

    @ti.kernel
    def draw_external_force(self, f:ti.i32):
        inv_rot_h1 = self.fem_sensor1.rot_h[None].inverse()
        inv_trans_h1 = self.fem_sensor1.inv_trans_h[None]
        half_seg = self.fem_sensor1.num_triangles #//2
        for i in range(half_seg):
            f_1 = self.fem_sensor1.external_force_field[f, self.fem_sensor1.contact_seg[i][0]]
            i_1 = self.fem_sensor1.virtual_pos[f, self.fem_sensor1.contact_seg[i][0]]
            ti_1 = inv_trans_h1 @ ti.Vector([i_1[0], i_1[1], i_1[2], 1.0])
            tf_1 = inv_rot_h1 @ f_1

            self.surf_offset1[i][0] = -1*tf_1[0] * 0.01 #* 0.01
            self.surf_offset1[i][1] = -1*tf_1[2] * 0.01 #* 0.01
            self.surf_init_pos1[i][0] = ti_1[0] + 0.5
            self.surf_init_pos1[i][1] = ti_1[2] + 0.5

    def draw_markers(self, init_markers, cur_markers, gui):
        img_height = 480
        img_width = 640
        scale = img_width
        rescale = 1.8
        draw_points = rescale * (init_markers - [320, 240]) / scale + [0.5, 0.5]
        offset = rescale * (cur_markers - init_markers) / scale
        if not off_screen:
            gui.circles(draw_points, radius=2, color=0xf542a1)
            gui.arrows(draw_points, 10.0*offset, radius=2, color=0xe6c949)
        


    def draw_markers_cv(self, tracked_markers, init_markers, showScale):
        img = np.zeros((480, 640, 3))
        markerCenter=np.around(init_markers[:,0:2]).astype(np.int16)
        for i in range(init_markers.shape[0]):
            marker_motion = tracked_markers[i] - init_markers[i]
            cv2.arrowedLine(img,(markerCenter[i,0], markerCenter[i,1]), \
                (int(init_markers[i,0]+marker_motion[0]*showScale), int(init_markers[i,1]+marker_motion[1]*showScale)),\
                (0, 255, 255),2)
        return img

    def apply_action(self, action, ts):
        d_pos1, d_ori1, d_pos2, d_ori2 = np.split(action, 4)
        self.fem_sensor1.d_pos.from_numpy(d_pos1)
        self.fem_sensor1.d_ori.from_numpy(d_ori1)
        self.fem_sensor1.set_pose_control()
        self.fem_sensor1.set_control_vel(0)
        self.fem_sensor1.set_vel(0)
        self.reset()
        for ss in range(self.sub_steps - 1):
            self.update(ss)

        self.memory_to_cache(ts)

    @ti.kernel
    def compute_angle(self, t: ti.i32):
        
        for f in range(self.sub_steps -1 ):
            if f == 0:
                self.angle[t+1] = self.angle[t] + ti.atan2(self.mpm_object.R[f][1,0], self.mpm_object.R[f][1,1])
            else:
                self.angle[t+1] += ti.atan2(self.mpm_object.R[f][1,0], self.mpm_object.R[f][1,1])


    @ti.kernel
    def compute_angle_loss(self,  t: ti.i32):
        self.loss[None] += self.alpha[None] * (self.angle[t] - self.target_angle[None])* (self.angle[t] - self.target_angle[None])
    
    
    @ti.kernel
    def draw_surface(self, f:ti.i32):
        inv_trans_h1 = self.fem_sensor1.inv_trans_h[None]
        half_seg = self.fem_sensor1.num_triangles #//2
        for i in range(half_seg):
            p_1 = self.fem_sensor1.pos[f, self.fem_sensor1.contact_seg[i][0]] # triangle's 1st node
            i_1 = self.fem_sensor1.init_x[self.fem_sensor1.contact_seg[i][0]]
            tp_1 = inv_trans_h1 @ ti.Vector([p_1[0], p_1[1], p_1[2], 1.0])


            self.surf_offset1[i][0] = 2*(tp_1[0] - i_1[0])
            self.surf_offset1[i][1] = 2*(tp_1[2] - i_1[2])
            self.press_offset1[i] = tp_1[1] - i_1[1]
            self.surf_init_pos1[i][0] = i_1[0] + 0.5
            self.surf_init_pos1[i][1] = i_1[2] + 0.5

    @ti.kernel
    def draw_perspective(self, f:ti.i32):
        phi, theta = ti.math.radians(self.view_phi), ti.math.radians(self.view_theta) # 28 20
        c_p, s_p = ti.math.cos(phi), ti.math.sin(phi)
        c_t, s_t = ti.math.cos(theta), ti.math.sin(theta)
        offset = 0.2
        for i in range(self.fem_sensor1.n_verts):
            x, y, z = self.fem_sensor1.pos[f, i][0] - offset, self.fem_sensor1.pos[f, i][1] - offset, self.fem_sensor1.pos[f, i][2] - offset
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            u, v = xx, y * c_t + zz * s_t
            self.draw_pos2[i][0] = u + 0.2
            self.draw_pos2[i][1] = v + 0.5

        for i in range(self.mpm_object.n_particles):
            x, y, z = self.mpm_object.x_0[f, i][0] - offset, self.mpm_object.x_0[f, i][1] - offset, self.mpm_object.x_0[f, i][2] - offset
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            u, v = xx, y * c_t + zz * s_t
            self.draw_pos3[i][0] = u + 0.2
            self.draw_pos3[i][1] = v + 0.5

    def draw_triangles(self, sensor, gui, f, tphi, ttheta, viz_scale, viz_offset):
        inv_trans_h = sensor.inv_trans_h[None]
        pos_ = sensor.pos.to_numpy()[f,:]
        init_pos_ = sensor.virtual_pos.to_numpy()[f,:]
        ones = np.ones((pos_.shape[0],1))

        hom_pos_ = np.hstack((pos_, ones)) # N x 4
        c_pos_ = np.matmul(inv_trans_h,hom_pos_.T).T[:,0:3] # Nx3

        hom_pos_ = np.hstack((init_pos_, ones)) # N x 4
        v_pos_ = np.matmul(inv_trans_h,hom_pos_.T).T[:,0:3] # Nx3

        phi, theta = np.radians(tphi), np.radians(ttheta) # 28 20
        c_p, s_p = np.cos(phi), np.sin(phi)
        c_t, s_t = np.cos(theta), np.sin(theta)

        c_seg_ = sensor.contact_seg.to_numpy()
        offset = 0.0

        a, b, c = pos_[c_seg_[:, 0]], pos_[c_seg_[:, 1]], pos_[c_seg_[:, 2]]
        x = a[:,0]; y = a[:,1]; z = a[:,2]
        xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
        ua, va = xx + 0.2, y * c_t + zz * s_t + 0.5

        x = b[:,0]; y = b[:,1]; z = b[:,2]
        xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
        ub, vb = xx + 0.2, y * c_t + zz * s_t + 0.5

        x = c[:,0]; y = c[:,1]; z = c[:,2]
        xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
        uc, vc = xx + 0.2, y * c_t + zz * s_t + 0.5

        pa, pb, pc = c_pos_[c_seg_[:, 0]], c_pos_[c_seg_[:, 1]], c_pos_[c_seg_[:, 2]]
        ba, bb, bc = v_pos_[c_seg_[:, 0]], v_pos_[c_seg_[:, 1]], v_pos_[c_seg_[:, 2]]
        oa, ob, oc = pa[:, 1] - ba[:, 1], pb[:, 1] - bb[:, 1], pc[:, 1] - bc[:, 1]

        k = -1 * (oa + ob + oc) * (1 / 3) * 1.0 # z deformation
        gb = 0.5
        # gui.triangles(np.array([a[:,0],a[:,2]]).T, np.array([b[:,0],b[:,2]]).T, np.array([c[:,0],c[:,2]]).T, color=ti.rgb_to_hex([k + gb, gb, gb]))
        gui.triangles(viz_scale*np.array([ua,va]).T + viz_offset, viz_scale*np.array([ub,vb]).T + viz_offset, viz_scale*np.array([uc,vc]).T + viz_offset, color=ti.rgb_to_hex([k + gb, gb, gb]))

    def calculate_force(self):
        self.fem_sensor1.get_external_force(self.fem_sensor1.sub_steps - 2)
        self.mpm_object.get_external_force(self.mpm_object.sub_steps - 2)
        self.compute_contact_force(self.sub_steps - 2)

    def render(self, gui1, gui2, gui3):
        viz_scale = 0.1
        viz_offset = [0.0, 0.0]
        self.fem_sensor1.extract_markers(0)
        init_2d = self.fem_sensor1.virtual_markers.to_numpy()
        marker_2d = self.fem_sensor1.predict_markers.to_numpy()
        self.draw_markers(init_2d, marker_2d, gui2)
        self.draw_perspective(0)
        gui1.circles(viz_scale * self.draw_pos3.to_numpy() + viz_offset, radius=2, color=0x039dfc)
        gui1.circles(viz_scale * self.draw_pos2.to_numpy() + viz_offset, radius=2, color=0xe6c949)
        # gui1.circles(viz_scale * self.draw_pos4.to_numpy() + viz_offset, radius=2, color=0xe6c949)
        self.draw_triangles(self.fem_sensor1, gui3, 0, 0, 90, viz_scale, viz_offset)
        gui1.show()
        gui2.show()
        gui3.show()
    
    def prepare_env(self):
        self.init_pos_control()
        self.load_target()
    
    def apply_action(self, action, ts):
        if ts < self.prepare_step:
            d_pos = np.array([action[0], action[1] + 1.5, action[2]])
            d_ori = np.array([action[3], action[4], action[5]])
        else:
            d_pos = np.array([action[0] + 1.0, action[1] + 0.5, action[2]])
            d_ori = np.array([action[3], action[4], action[5]])
        # print(action)
        self.fem_sensor1.d_pos.from_numpy(d_pos)
        self.fem_sensor1.d_ori.from_numpy(d_ori)
        self.fem_sensor1.set_pose_control()
        self.fem_sensor1.set_control_vel(0)
        self.fem_sensor1.set_vel(0)
        self.reset()
        for ss in range(self.sub_steps - 1):
            self.update(ss)
        self.memory_to_cache(ts)
    
    def compute_loss(self, ts):
        if self.use_tactile:
            self.compute_force_loss(ts)
        if self.use_state:
            self.compute_angle_loss(ts+1)
        
        
    def calculate_force(self, ts):
        if self.use_tactile:
            self.compute_contact_force(self.sub_steps - 2)
        if self.use_state:
            self.compute_angle(ts)
    


def transform_2d(point, angle, translate):
    theta = np.radians(angle)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    new_point = np.matmul(rot_mat, point.T).T + translate
    return new_point


def main():
    ti.init(arch=ti.gpu, device_memory_GB=4)

    obj_name = "block-10.stl"
    num_sub_steps = 50
    num_total_steps = 600
    num_opt_steps = 100
    dt = 5e-5
    contact_model = Contact(use_tactile=USE_TACTILE, USE_STATE=USE_STATE, dt=dt, total_steps = num_total_steps, sub_steps = num_sub_steps,  obj=obj_name)

    if not off_screen:
        gui1 = ti.GUI("Contact Viz")
        gui2 = ti.GUI("Force Map 1")
        # gui3 = ti.GUI("Deformation Map 1")

    losses = []
    contact_model.init_pos_control()
    contact_model.load_target()
    # contact_model.load_pos_control()
   
    # pred_force = []
    form_loss = 0

    for opts in range(0, num_opt_steps):
        print("Opt # step ======================", opts)
        contact_model.init()
        contact_model.clear_all_grad()
        contact_model.clear_state_loss_grad()
        
        for ts in range(num_total_steps-1):
            contact_model.iter = ts
            contact_model.set_pos_control(ts)
            contact_model.fem_sensor1.set_pose_control()
            contact_model.fem_sensor1.set_control_vel(0)
            contact_model.fem_sensor1.set_vel(0)                      
            contact_model.reset()
            for ss in range(num_sub_steps - 1):
                contact_model.update(ss)
            
            contact_model.memory_to_cache(ts)
                

            # ########
            print("# FP Iter ", ts)

            if USE_TACTILE:
                contact_model.compute_contact_force(num_sub_steps - 2)
                form_loss = contact_model.loss[None]
                contact_model.compute_force_loss(ts)
                print("contact force: ", contact_model.predict_force1[ts])
                print("force loss", contact_model.loss[None]-form_loss)

            if USE_STATE:
                contact_model.compute_angle(ts)
                print("angle",  contact_model.angle[ts])

            ## visualizationw
            viz_scale = 0.1
            viz_offset = [0.0, 0.0]
            
            if not off_screen:
                contact_model.fem_sensor1.extract_markers(0)
                init_2d = contact_model.fem_sensor1.virtual_markers.to_numpy()
                marker_2d = contact_model.fem_sensor1.predict_markers.to_numpy()
                contact_model.draw_markers(init_2d, marker_2d, gui2)

            ### the external force is not propogate to the last time step but the second last
            # contact_model.draw_external_force(contact_model.fem_sensor1.sub_steps-2)
            if not off_screen:
                contact_model.draw_perspective(0)
                gui1.circles(viz_scale * contact_model.draw_pos3.to_numpy() + viz_offset, radius=2, color=0x039dfc)
                gui1.circles(viz_scale * contact_model.draw_pos2.to_numpy() + viz_offset, radius=2, color=0xe6c949)
                gui1.show()
                gui2.show()
                # gui3.show()
        ## backward!    
        loss_frame = 0
        form_loss = 0

        for ts in range(num_total_steps-2, -1, -1):
            print("BP", ts)
            contact_model.clear_all_grad()
     
            if USE_TACTILE:
                contact_model.compute_contact_force(num_sub_steps - 2)
                form_loss = contact_model.loss[None]
                contact_model.compute_force_loss(ts)
                print("force loss", contact_model.loss[None]-form_loss)

            if USE_STATE:
                form_loss = contact_model.loss[None]
                contact_model.compute_angle_loss(ts+1)
                print("angle loss", contact_model.loss[None]-form_loss)
                contact_model.compute_angle_loss.grad(ts+1)
                
                contact_model.compute_angle.grad(ts)
               
            if USE_TACTILE:
                contact_model.compute_force_loss.grad(ts)
                contact_model.compute_contact_force.grad(num_sub_steps - 2)
            
            for ss in range(num_sub_steps-2, -1, -1):
                contact_model.update_grad(ss)



            contact_model.fem_sensor1.set_vel.grad(0)
            contact_model.fem_sensor1.set_control_vel.grad(0)
            contact_model.fem_sensor1.set_pose_control.grad()


            contact_model.set_pos_control.grad(ts)
            
            ### optimization with grad backpropagation        
            grad_p1 = contact_model.p_sensor1.grad[ts]
            grad_o1 = contact_model.o_sensor1.grad[ts]
     

            lr_p = 0.5e1
            lr_o = 1e3
            contact_model.p_sensor1[ts] -= lr_p * grad_p1
            contact_model.o_sensor1[ts] -= lr_o * grad_o1
     

            loss_frame += contact_model.loss[None]
            print("# BP Iter: ", ts, " loss: ", contact_model.loss[None])
            print("P/O grads: ", grad_p1, grad_o1)
            print("P/O updated: ", contact_model.p_sensor1[ts], contact_model.o_sensor1[ts])

            ### rerun the forward one more time to fill the grad
            if (ts - 1) >=0:
                contact_model.memory_from_cache(ts-1)
                contact_model.set_pos_control(ts-1)
                contact_model.fem_sensor1.set_pose_control_bp()
                contact_model.fem_sensor1.set_control_vel(0)
                contact_model.fem_sensor1.set_vel(0)
                contact_model.reset()
                for ss in range(num_sub_steps - 1):
                    contact_model.update(ss)

            if not off_screen:
                contact_model.fem_sensor1.extract_markers(0)
                init_2d = contact_model.fem_sensor1.virtual_markers.to_numpy()
                marker_2d = contact_model.fem_sensor1.predict_markers.to_numpy()
                contact_model.draw_markers(init_2d, marker_2d, gui2)
                
                contact_model.draw_perspective(0)
                gui1.circles(viz_scale * contact_model.draw_pos3.to_numpy() + viz_offset, radius=2, color=0x039dfc)
                gui1.circles(viz_scale * contact_model.draw_pos2.to_numpy() + viz_offset, radius=2, color=0xe6c949)           

                gui1.show()
                gui2.show()
                # gui3.show()

        losses.append(loss_frame)


        if not os.path.exists(f"lr_object_repose_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}"):
            os.mkdir(f"lr_object_repose_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}")

        if not os.path.exists(f"results"):
            os.mkdir(f"results")


        ## save loss plot
        if opts % 5 == 0 or opts == num_opt_steps-1:
            print("# Iter ", opts, "Opt step loss: ", loss_frame)
            plt.title("Trajectory Optimization")
            plt.ylabel("Loss")
            plt.xlabel("Iter") # "Gradient Descent Iterations"
            plt.plot(losses)
            plt.savefig(os.path.join(f"lr_object_repose_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}",f"object_repose_state_{args.use_state}_tactile_{args.use_tactile}_{opts}.png"))
            np.save(os.path.join(f"lr_object_repose_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}", f"control_pos_{opts}.npy"), contact_model.p_sensor1.to_numpy())
            np.save(os.path.join(f"lr_object_repose_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}", f"control_ori_{opts}.npy"), contact_model.o_sensor1.to_numpy())
            np.save(os.path.join(f"lr_object_repose_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}", f"losses_{opts}.npy"), np.array(losses))
        
        ## save traj
        if loss_frame <= np.min(losses):
            best_p = contact_model.p_sensor1.to_numpy()
            best_o = contact_model.o_sensor1.to_numpy()
            np.save(os.path.join(f"lr_object_repose_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}","control_pos_best.npy"), best_p)
            np.save(os.path.join(f"lr_object_repose_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}","control_ori_best.npy"), best_o)
            print("Best traj saved!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_state", action = "store_true", help = "whether to use state loss")
    parser.add_argument("--use_tactile", action = "store_true", help = "whether to use tactile loss")



    parser.add_argument("--times", type = int, default = 1)

    args = parser.parse_args()
    USE_STATE = args.use_state
    USE_TACTILE = args.use_tactile
    main()
