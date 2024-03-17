"""
a cable straightening task
"""

import taichi as ti
from math import pi
import numpy as np
import sys
from difftactile.object_model.PBD_rope  import PBDRope
from difftactile.sensor_model.gripper_kinematics import Gripper
import os
import cv2
import math
import trimesh
# fem: cm
import matplotlib.pyplot as plt


Off_screen = False




@ti.data_oriented
class Contact:
    def __init__(self, use_state, use_tactile, dt=5e-5, total_steps=300, sub_steps = 50, obj=None):
        self.dt = dt
        self.total_steps = total_steps
        self.sub_steps = sub_steps
        self.press_steps = self.total_steps//4
        self.dim = 3

        self.p_particles = 50
        self.p_rad = 0.25
        self.table_height = 0.0
        self.use_state = use_state
        self.use_tactile = use_tactile
        self.rope_object = PBDRope(dt=dt,
                                    sub_steps=sub_steps,
                                    p_rad=self.p_rad,
                                    p_rho = 10.0,
                                    n_particles = self.p_particles,
                                    table_height = self.table_height,
                                    rest_length = 0.1)





        data_path = os.path.join("..", "meshes", "sensors")

        self.view_phi = 90
        self.view_theta = 0
        self.view_scale = 10.0
        self.table_scale = 2.0
        self.num_sensor = 2

        self.gripper = Gripper(data_path, sub_steps=sub_steps, mesh_vis_scale = self.view_scale)

        self.kn = ti.field(dtype=float, shape=(), needs_grad=True)
        self.kd = ti.field(dtype=float, shape=(), needs_grad=True)
        self.kt = ti.field(dtype=float, shape=(), needs_grad=True)
        self.friction_coeff = ti.field(dtype=float, shape=(), needs_grad=True)

        self.kn[None] = 55.33#180.0#80.0
        self.kd[None] = 239.97#120.0
        self.kt[None] = 94.35#10.0 #5.0
        self.friction_coeff[None] = 4.9#1.6#0.8


        self.gripper.fem_sensor1.mu[None] = 1294.01
        self.gripper.fem_sensor1.lam[None] = 9201.11
        self.gripper.fem_sensor2.mu[None] = 1294.01
        self.gripper.fem_sensor2.lam[None] = 9201.11

        self.contact_idx = ti.Vector.field(self.num_sensor, dtype=int, shape=(self.sub_steps, self.rope_object.n_particles)) # indicating the contact seg idx of fem sensor for closested triangle mesh
        self.contact_flag = ti.field(dtype=int, shape=(self.sub_steps, self.rope_object.n_particles))
        self.total_ext_f = ti.Vector.field(3, dtype=float, shape=())


        # Transloation, Rotation and Width speed of the gripper
        self.p_gripper = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.total_steps), needs_grad=True)
        self.o_gripper = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.total_steps), needs_grad=True)
        self.w_gripper = ti.field(dtype=ti.f32,shape=(self.total_steps), needs_grad=True )



        self.loss = ti.field(float, (), needs_grad=True)


        ## Contact force
        self.target_force1 = ti.Vector.field(self.dim, float, shape=())
        self.target_force2 = ti.Vector.field(self.dim, float, shape=())
        self.predict_force1 = ti.Vector.field(self.dim, float, shape=(), needs_grad=True)
        self.predict_force2 = ti.Vector.field(self.dim, float, shape=(), needs_grad=True)
        self.contact_force1 = ti.Vector.field(self.dim, float, shape=(), needs_grad=True)
        self.contact_force2 = ti.Vector.field(self.dim, float, shape=(), needs_grad=True)


        ## Contact loc
        self.target_loc1 = ti.Vector.field(self.dim, float, shape=())
        self.target_loc2 = ti.Vector.field(self.dim, float, shape=())
        self.predict_loc1 = ti.Vector.field(self.dim, float, shape = (), needs_grad=True)
        self.predict_loc2 = ti.Vector.field(self.dim, float, shape = (), needs_grad=True)

        self.wf = 1e-3
        self.wl = 1.0
        self.alpha = 1e-2
        self.beta = 1e-5

        self.norm_eps = 1e-11

        ## 2D visualization
        self.surf_offset1 = ti.Vector.field(2, float, self.gripper.fem_sensor1.num_triangles)
        self.surf_init_pos1 = ti.Vector.field(2, float, self.gripper.fem_sensor1.num_triangles)
        self.press_offset1 = ti.field(float, self.gripper.fem_sensor1.num_triangles)

        self.draw_pos1 = ti.Vector.field(2, float, self.gripper.fem_sensor1.n_verts) # elastomer1's pos
        self.draw_pos2 = ti.Vector.field(2, float, self.gripper.fem_sensor2.n_verts) # elastomer2's pos
        self.draw_pos = ti.Vector.field(2, dtype=float, shape=(self.rope_object.n_particles)) # rope's pos
        self.draw_tableline = ti.Vector.field(3, dtype=float, shape=(2*4)) # table

        # 3d viz
        self.draw_pos_3d = ti.Vector.field(3, dtype=float, shape=(self.rope_object.n_particles))
        self.draw_fem1_3d = ti.Vector.field(3, dtype=float, shape=(self.gripper.fem_sensor1.n_verts))
        self.draw_fem2_3d = ti.Vector.field(3, dtype=float, shape=(self.gripper.fem_sensor2.n_verts))
        self.color_fem1_3d = ti.Vector.field(3, dtype=float, shape=(self.gripper.fem_sensor1.n_verts))
        self.color_fem2_3d = ti.Vector.field(3, dtype=float, shape=(self.gripper.fem_sensor2.n_verts))

    def init(self):

        self.init_pos = [0.0, 2.0, -2.0]
        self.init_ori = [0.0, 0.0, 1.0]
        self.rope_object.init(self.init_pos, self.init_ori)
        self.rope_object.set_target(self.init_pos)


        ### extract the height of the obj
        obj_pos = self.rope_object.pos.to_numpy()[0,:]
        sensor_pos = self.gripper.fem_sensor1.init_x.to_numpy()
        print("Obj height: ", np.max(obj_pos[:,1]), np.min(obj_pos[:,1]))
        print("Sensor height: ", np.max(sensor_pos[:,1]), np.min(sensor_pos[:,1]))
        self.init_h = np.max(obj_pos[:,1])
        print("Init height: ", self.init_h)


        rot = [0.0, 0.0, 0.0]
        gripper_ee_hight = self.init_h + 9.3
        t = [0.0, gripper_ee_hight, -2.0]
        grip = 5.0
        self.gripper.init(rot, t, grip)
        g_trans_h = self.gripper.ee_delta[0].to_numpy()
        g_trans_h[1,3] -= 2.0
        self.gripper.gripper_base.update(g_trans_h)
        g_trans_finger1 = self.gripper.finger1_delta[0].to_numpy()
        g_trans_finger2 = self.gripper.finger2_delta[0].to_numpy()
        self.gripper.gripper_finger1.update(g_trans_finger1)
        self.gripper.gripper_finger2.update(g_trans_finger2)
        self.gripper.init_fem_sensor()

        self.in_contact = False
        self.contact_timestamp = 0



    def update(self, f):

        self.gripper.fem_sensor1.update(f)
        self.gripper.fem_sensor2.update(f)
        self.check_collision(f)
        self.collision(f)
        self.rope_object.update(f)
        self.gripper.fem_sensor1.update2(f)
        self.gripper.fem_sensor2.update2(f)


    def update_grad(self, f):
        self.gripper.fem_sensor2.update2.grad(f)
        self.gripper.fem_sensor2.update2.grad(f)
        self.rope_object.update.grad(f)
        self.clamp_grid(f)
        self.collision.grad(f)
        self.gripper.fem_sensor2.update.grad(f)
        self.gripper.fem_sensor1.update.grad(f)



    @ti.kernel
    def set_pos_control(self, f:ti.i32):
        self.gripper.d_pos[None] = self.p_gripper[f]
        self.gripper.d_ori[None] = self.o_gripper[f]
        self.gripper.d_gripper[None] = self.w_gripper[f]


    @ti.kernel
    def clamp_grid(self, f:ti.i32):
        for p in range(self.rope_object.n_particles):
            self.rope_object.pos.grad[f, p] = ti.math.clamp(self.rope_object.pos.grad[f, p], -1000.0, 1000.0)
            self.rope_object.vel.grad[f, p] = ti.math.clamp(self.rope_object.vel.grad[f, p], -1000.0, 1000.0)

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
        normal_vel_p0 = norm_v.dot(relative_vel_p0)
        normal_factor_p0 = -(self.kn[None] + self.kd[None] * normal_vel_p0)* sdf * norm_v
        shear_vel_p0 = relative_vel_p0 - normal_vel_p0 * norm_v
        shear_vel_norm_p0 = shear_vel_p0.norm()
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
    def compute_contact_loc(self, f:ti.i32):
        for i in range(self.gripper.fem_sensor1.num_surface):
            idp = self.gripper.fem_sensor1.surface_id[i]
            cur_pos = self.gripper.fem_sensor1.pos[f, idp]
            inv_pos = self.gripper.fem_sensor1.inv_trans_h[None] @ ti.Vector([cur_pos[0], cur_pos[1], cur_pos[2], 1.0])
            pos = ti.Vector([inv_pos[0], inv_pos[1], inv_pos[2]])
            ext_f = self.gripper.fem_sensor1.external_force_field[f, idp]
            if ext_f.norm(1e-10) > 1e-4:
                self.predict_loc1[None] += pos

        for i in range(self.gripper.fem_sensor2.num_surface):
            idp = self.gripper.fem_sensor1.surface_id[i]
            cur_pos = self.gripper.fem_sensor2.pos[f, idp]
            inv_pos = self.gripper.fem_sensor2.inv_trans_h[None] @ ti.Vector([cur_pos[0], cur_pos[1], cur_pos[2], 1.0])
            pos = ti.Vector([inv_pos[0], inv_pos[1], inv_pos[2]])
            ext_f = self.gripper.fem_sensor2.external_force_field[f, idp]
            if ext_f.norm(1e-10) > 1e-4:
                self.predict_loc2[None] += pos


    @ti.kernel
    def compute_force_loss(self):
        self.predict_force1[None] = self.gripper.fem_sensor1.inv_rot[None] @ self.contact_force1[None]
        self.predict_force2[None] = self.gripper.fem_sensor2.inv_rot[None] @ self.contact_force2[None]

        self.loss[None] += self.beta*self.wf*((self.predict_force1[None][1] - self.target_force1[None][1])**2 + (self.predict_force1[None][0] - self.target_force1[None][0])**2)
        self.loss[None] += self.beta*self.wf*((self.predict_force2[None][1] - self.target_force2[None][1])**2 + (self.predict_force2[None][0] - self.target_force2[None][0])**2)

    @ti.kernel
    def compute_loc_loss(self):
        self.loss[None] += self.beta*self.wl*((self.target_loc1[None][0] - self.predict_loc1[None][0])**2 + (self.target_loc1[None][2] - self.predict_loc1[None][2])**2)
        self.loss[None] += self.beta*self.wl*((self.target_loc2[None][0] - self.predict_loc2[None][0])**2 + (self.target_loc2[None][2] - self.predict_loc2[None][2])**2)


    @ti.kernel
    def compute_travel_loss(self):
        for p in  range(self.rope_object.n_particles):
            self.loss[None] -= (self.rope_object.travel_info[self.sub_steps-2, p][0] )

    @ti.kernel
    def compute_rope_pos_loss(self, f:ti.i32):
        for p in range(self.rope_object.n_particles):
            self.loss[None] += self.alpha*(self.rope_object.pos[f, p][0] - self.rope_object.init_pos[p][0])**2 + (self.rope_object.pos[f, p][1] - self.rope_object.init_pos[p][1])**2 + \
            (self.rope_object.pos[f, p][2] - self.rope_object.init_pos[p][2])**2

    @ti.kernel
    def check_collision(self, f:ti.i32):
        for p in range(self.rope_object.n_particles):
            cur_p = self.rope_object.pos[f, p]
            min_idx1 = self.gripper.fem_sensor1.find_closest(cur_p, f)
            min_idx2 = self.gripper.fem_sensor2.find_closest(cur_p, f)
            self.contact_idx[f, p] = [min_idx1, min_idx2]

    @ti.kernel
    def collision(self, f:ti.i32):
        for p in range(self.rope_object.n_particles):
            ### boundary condition with sensor elastomer sdf
            # use the center of the grid to check the sdf
            cur_p = self.rope_object.pos[f, p]
            cur_v = self.rope_object.vel[f, p]
            min_idx1, min_idx2 = self.contact_idx[f, p]
            cur_sdf1, cur_norm_v1, cur_relative_v1, contact_flag1 = self.gripper.fem_sensor1.find_sdf(cur_p, cur_v, min_idx1, f, self.rope_object.p_rad)
            cur_sdf2, cur_norm_v2, cur_relative_v2, contact_flag2 = self.gripper.fem_sensor2.find_sdf(cur_p, cur_v, min_idx2, f, self.rope_object.p_rad)
            if contact_flag1:
                ext_v1, ext_n1, ext_t1 = self.calculate_contact_force(cur_sdf1, -1*cur_norm_v1, -1*cur_relative_v1)
                self.rope_object.update_contact_force(ext_v1, f, p)
                self.gripper.fem_sensor1.update_contact_force(min_idx1, -1*ext_v1, f)
                self.contact_flag[f, p] = 1
            if contact_flag2:
                ext_v2, ext_n2, ext_t2 = self.calculate_contact_force(cur_sdf2, -1*cur_norm_v2, -1*cur_relative_v2)
                self.rope_object.update_contact_force(ext_v2, f, p)
                self.gripper.fem_sensor2.update_contact_force(min_idx2, -1*ext_v2, f)

            if contact_flag1 or contact_flag2:
                self.rope_object.travel_info[f+1, p][0] = 1.0

            else:
                self.rope_object.travel_info[f+1, p] = self.rope_object.travel_info[f, p]


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
        ### to do double check the number here!!
        self.target_force1[None] = ti.Vector([-300.0, -300.0, 0.0]) # maybe don't penalize fx
        self.target_force2[None] = ti.Vector([300.0, -300.0, 0.0])
        self.target_loc1[None] = ti.Vector([0.0, 0.0, 0.0])
        self.target_loc2[None] = ti.Vector([0.0, 0.0, 0.0])



    @ti.kernel
    def clear_loss_grad(self):
        self.kn.grad[None] = 0.0
        self.kd.grad[None] = 0.0
        self.kt.grad[None] = 0.0
        self.friction_coeff.grad[None] = 0.0

        self.loss[None] = 0.0
        self.loss.grad[None] = 1.0

        self.contact_force1.grad[None].fill(0.0)
        self.contact_force2.grad[None].fill(0.0)
        self.predict_force1.grad[None].fill(0.0)
        self.predict_force2.grad[None].fill(0.0)
        self.predict_loc1.grad[None].fill(0.0)
        self.predict_loc2.grad[None].fill(0.0)


        self.p_gripper.grad.fill(0.0)
        self.o_gripper.grad.fill(0.0)
        self.w_gripper.grad.fill(0.0)




    def clear_traj_grad(self):
        self.gripper.clear_loss_grad()
        self.gripper.fem_sensor1.clear_loss_grad()
        self.gripper.fem_sensor2.clear_loss_grad()
        self.rope_object.clear_loss_grad()
        self.clear_loss_grad()

    def clear_all_grad(self):
        self.clear_traj_grad()
        self.gripper.clear_step_grad(self.sub_steps)
        self.gripper.fem_sensor1.clear_step_grad(self.sub_steps)
        self.gripper.fem_sensor2.clear_step_grad(self.sub_steps)
        self.rope_object.clear_step_grad(self.sub_steps)

    def clear_step_grad(self):
        self.clear_traj_grad()
        self.gripper.clear_step_grad(self.sub_steps-1)
        self.gripper.fem_sensor1.clear_step_grad(self.sub_steps-1)
        self.gripper.fem_sensor2.clear_step_grad(self.sub_steps-1)
        self.rope_object.clear_step_grad(self.sub_steps-1)



    def reset(self):
        self.rope_object.reset()
        self.gripper.fem_sensor1.reset_contact()
        self.gripper.fem_sensor2.reset_contact()
        self.contact_idx.fill(-1)
        self.contact_flag.fill(0)
        self.predict_force1[None].fill(0.0)
        self.predict_force2[None].fill(0.0)
        self.contact_force1[None].fill(0.0)
        self.contact_force2[None].fill(0.0)
        self.predict_loc1[None].fill(0.0)
        self.predict_loc2[None].fill(0.0)

    def memory_to_cache(self, t):
        self.gripper.memory_to_cache(t)
        self.gripper.fem_sensor1.memory_to_cache(t)
        self.gripper.fem_sensor2.memory_to_cache(t)
        self.rope_object.memory_to_cache(t)

    def memory_from_cache(self, t):
        self.gripper.memory_from_cache(t)
        self.gripper.fem_sensor1.memory_from_cache(t)
        self.gripper.fem_sensor2.memory_from_cache(t)
        self.rope_object.memory_from_cache(t)




    @ti.kernel
    def draw_external_force(self, f:ti.i32):
        inv_rot_h1 = self.gripper.fem_sensor1.rot_h[None].inverse()
        inv_trans_h1 = self.gripper.fem_sensor1.trans_h[None].inverse()
        half_seg = self.gripper.fem_sensor1.num_triangles #//2
        for i in range(half_seg):
            f_1 = self.gripper.fem_sensor1.external_force_field[f, self.gripper.fem_sensor1.contact_seg[i][0]]
            i_1 = self.gripper.fem_sensor1.virtual_pos[f, self.gripper.fem_sensor1.contact_seg[i][0]]
            ti_1 = inv_trans_h1 @ ti.Vector([i_1[0], i_1[1], i_1[2], 1.0])
            tf_1 = inv_rot_h1 @ f_1

            self.surf_offset1[i][0] = -1*tf_1[0] * 0.01 #* 0.01
            self.surf_offset1[i][1] = -1*tf_1[2] * 0.01 #* 0.01
            self.surf_init_pos1[i][0] = ti_1[0] + 0.5
            self.surf_init_pos1[i][1] = ti_1[2] + 0.5


    @ti.kernel
    def init_control_parameters(self):

        vx = 0.0; vy = 0.0; vz = 0.0
        rx = 0.0; ry = 0.0; rz = 0.0
        ws = 0.8   # ws > 0, gripper width will get shorter

        for i in range(0, self.press_steps):
            self.p_gripper[i] = ti.Vector([vx, vy, vz])
            self.o_gripper[i] = ti.Vector([rx, ry, rz])
            self.w_gripper[i] = ws

        vx = 0.0; vy = 0.0; vz = 0.8
        rx = 0.0; ry = 0.0; rz = 0.0
        ws = 0.0  # ws > 0, gripper width will get shorter

        for i in range(self.press_steps, self.total_steps):
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


    @ti.kernel
    def draw_surface(self, f:ti.i32):
        inv_trans_h1 = self.gripper.fem_sensor1.trans_h[None].inverse()
        half_seg = self.gripper.fem_sensor1.num_triangles #//2
        for i in range(half_seg):
            p_1 = self.gripper.fem_sensor1.pos[f, self.gripper.fem_sensor1.contact_seg[i][0]] # triangle's 1st node
            # b_1 = self.fem_sensor1.pos[f, self.fem_sensor1.base_seg[i][0]] # triangle's 1st node
            i_1 = self.gripper.fem_sensor1.init_x[self.gripper.fem_sensor1.contact_seg[i][0]]

            tp_1 = inv_trans_h1 @ ti.Vector([p_1[0], p_1[1], p_1[2], 1.0])
            # tb_1 = inv_trans_h1 @ ti.Vector([b_1[0], b_1[1], b_1[2], 1.0])

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
        for i in range(self.gripper.fem_sensor1.n_verts):
            x, y, z = self.gripper.fem_sensor1.pos[f, i][0] - offset, self.gripper.fem_sensor1.pos[f, i][1] - offset, self.gripper.fem_sensor1.pos[f, i][2] - offset
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            u, v = xx, y * c_t + zz * s_t
            self.draw_pos1[i][0] = u + 0.2
            self.draw_pos1[i][1] = v + 0.5
        for i in range(self.gripper.fem_sensor2.n_verts):
            x, y, z = self.gripper.fem_sensor2.pos[f, i][0] - offset, self.gripper.fem_sensor2.pos[f, i][1] - offset, self.gripper.fem_sensor2.pos[f, i][2] - offset
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            u, v = xx, y * c_t + zz * s_t
            self.draw_pos2[i][0] = u + 0.2
            self.draw_pos2[i][1] = v + 0.5
        for i in range(self.rope_object.n_particles):
            x, y, z = self.rope_object.pos[f, i][0] - offset, self.rope_object.pos[f, i][1] - offset, self.rope_object.pos[f, i][2] - offset
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            u, v = xx, y * c_t + zz * s_t
            self.draw_pos[i][0] = u + 0.2
            self.draw_pos[i][1] = v + 0.5

    def draw_triangles(self, sensor, gui, f, tphi, ttheta, viz_scale, viz_offset):
        inv_trans_h = sensor.trans_h[None].inverse()
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

        a, b, c = c_pos_[c_seg_[:, 0]], c_pos_[c_seg_[:, 1]], c_pos_[c_seg_[:, 2]]
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

        ext_f = sensor.external_force_field.to_numpy()[f,:]
        in_contact_flag = np.sum(np.abs(ext_f), axis=1) > 0
        if np.sum(in_contact_flag) > 0:
            in_c_pos = c_pos_[in_contact_flag,:]

            x = in_c_pos[:,0]; y = in_c_pos[:,1]; z = in_c_pos[:,2]
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            ui, vi = xx + 0.2, y * c_t + zz * s_t + 0.5

            avg_pos = np.mean(in_c_pos, axis=0)
            x = avg_pos[0]; y = avg_pos[1]; z = avg_pos[2]
            xx, zz = x * c_p + z * s_p, z * c_p - x * s_p
            ua, va = xx + 0.2, y * c_t + zz * s_t + 0.5

            gui.circles(viz_scale*np.array([ui,vi]).T  + viz_offset, radius=2, color=0xf542a1)
            gui.circle(viz_scale*np.array([ua,va]).T  + viz_offset, radius=5, color=0xe6c949)

    def draw_table(self):

        c1 = ti.Vector([-self.table_scale, self.table_height, -self.table_scale])
        c2 = ti.Vector([-self.table_scale, self.table_height, self.table_scale])
        c3 = ti.Vector([self.table_scale, self.table_height, self.table_scale])
        c4 = ti.Vector([self.table_scale, self.table_height, -self.table_scale])
        self.draw_tableline[0] = c1; self.draw_tableline[1] = c2
        self.draw_tableline[2] = c2; self.draw_tableline[3] = c3
        self.draw_tableline[4] = c3; self.draw_tableline[5] = c4
        self.draw_tableline[6] = c4; self.draw_tableline[7] = c1

    @ti.kernel
    def draw_3d_scene(self, f:ti.i32):
        for p in range(self.rope_object.n_particles):
            self.draw_pos_3d[p] = self.rope_object.pos[f, p] / self.view_scale

        # only draw surface
        for p in range(self.gripper.fem_sensor1.num_surface):
            idx = self.gripper.fem_sensor1.surface_id[p]
            self.draw_fem1_3d[p] = self.gripper.fem_sensor1.pos[f, idx] / self.view_scale
            self.color_fem1_3d[p] = ti.Vector([0.9, 0.7, 0.8]) + 2.0*(self.gripper.fem_sensor1.pos[f, idx] - self.gripper.fem_sensor1.virtual_pos[f, idx])


        for p in range(self.gripper.fem_sensor2.num_surface):
            idx = self.gripper.fem_sensor2.surface_id[p]
            self.draw_fem2_3d[p] = self.gripper.fem_sensor2.pos[f, idx] / self.view_scale
            self.color_fem2_3d[p] = ti.Vector([0.02, 0.8, 0.9]) + 2.0*(self.gripper.fem_sensor2.pos[f, idx] - self.gripper.fem_sensor2.virtual_pos[f, idx])

    def scale_mesh_visualize(self):
        self.gripper.gripper_base.scale_visualize()
        self.gripper.gripper_finger1.scale_visualize()
        self.gripper.gripper_finger2.scale_visualize()

    def apply_action(self, action, ts):
        if ts < self.press_steps:
            d_pos = np.array([action[0], action[1], action[2]])
            d_ori = np.array([action[3], action[4], action[5]])
            d_wid = 0.8 + action[6]
        else:
            d_pos = np.array([action[0], action[1], action[2]+0.8])
            d_ori = np.array([action[3], action[4], action[5]])
            d_wid = action[6]
        self.gripper.d_pos.from_numpy(d_pos)
        self.gripper.d_ori.from_numpy(d_ori)
        self.gripper.d_gripper[None] = d_wid

        # self.set_pos_control(ts)
        for ss in range(self.sub_steps - 1):
            self.gripper.kinematic(ss)
        self.gripper.set_sensor_vel(0)
        self.gripper.set_sensor_pos(self.sub_steps - 1)
        self.reset()
        for ss in range(self.sub_steps - 1):
            self.update(ss)
        self.memory_to_cache(ts)
    
    def prepare_env(self):
        self.init_control_parameters()
        self.load_target()

    def calculate_force(self, ts):
        if self.use_tactile:
            self.compute_contact_force(self.sub_steps - 2)
            self.compute_contact_loc(self.sub_steps - 2)

    def compute_loss(self, ts):
        if self.use_tactile:
            self.compute_force_loss()
            self.compute_loc_loss()
        if self.use_state:
            self.compute_rope_pos_loss(self.sub_steps - 2)


def transform_2d(point, angle, translate):
    theta = np.radians(angle)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

    new_point = np.matmul(rot_mat, point.T).T + translate

    return new_point

def extract_translation_and_euler_angles(matrix):
    translation = matrix[:3, 3]
    rotation_matrix = matrix[:3, :3]
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        x = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y = math.atan2(-rotation_matrix[2, 0], sy)
        z = 0

    x_deg = math.degrees(x)
    y_deg = math.degrees(y)
    z_deg = math.degrees(z)

    return translation, (x_deg, y_deg, z_deg)


def hasnan(vector):
    for i in range(3):
        if math.isnan(vector[i]):
            return True
        else:
            return False


def main():
    ti.init(arch=ti.gpu, device_memory_GB=4)
    if not Off_screen:

        window = ti.ui.Window("Rope manipulation" , (512, 512))
        canvas = window.get_canvas()
        canvas.set_background_color((1, 1, 1))
        scene = ti.ui.Scene()
        camera = ti.ui.Camera()
        camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        camera.position(-0.1, 0.5, 0.5)
        camera.up(0, 1, 0)
        camera.lookat(0.0, 0.1, 0.0)
        camera.fov(75)
        gui1 = ti.GUI("Contact Viz")
        gui2 = ti.GUI("Force Map 1")
        gui3 = ti.GUI("Deformation Map 1")

    num_sub_steps = 50
    num_total_steps = 800
    num_opt_steps = 200
    dt = 5e-4
    contact_model = Contact(use_tactile=USE_TACTILE, use_state=USE_STATE, dt=dt, total_steps = num_total_steps, sub_steps = num_sub_steps)

    contact_model.draw_table()
    contact_model.init_control_parameters()
    contact_model.load_target()

    form_loss = 0
    losses = []

    for opts in range(num_opt_steps):
        print("Opt # step ======================", opts)
        contact_model.init()
        contact_model.clear_all_grad()


        for ts in range(num_total_steps-1):

            print("FP", ts)
            contact_model.set_pos_control(ts)
            for ss in range(num_sub_steps-1):
                contact_model.gripper.kinematic(ss)
                contact_model.update_visualization(ss+1)

            contact_model.gripper.set_sensor_vel(0)
            contact_model.gripper.set_sensor_pos(num_sub_steps-1)
            contact_model.reset()
            for ss in range(num_sub_steps-1):
                contact_model.update(ss)
            contact_model.memory_to_cache(ts)
            print("# FP Iter ", ts)
            form_loss = contact_model.loss[None]
            contact_model.compute_contact_force(num_sub_steps - 2)
            contact_model.compute_force_loss()
            print("force loss", contact_model.loss[None] - form_loss)
            print("Predict force 1/2", contact_model.predict_force1[None], contact_model.predict_force2[None])

            form_loss = contact_model.loss[None]
            contact_model.compute_contact_loc(num_sub_steps - 2)
            contact_model.compute_loc_loss()
            print("loc loss", contact_model.loss[None] - form_loss)
            print("Predict loc 1/2", contact_model.predict_loc1[None], contact_model.predict_loc2[None])

            rope_pos = contact_model.rope_object.pos.to_numpy()[0,:]
            print("Rope pos max/min: ", np.max(rope_pos), np.min(rope_pos))

            form_loss = contact_model.loss[None]
            contact_model.compute_rope_pos_loss(num_sub_steps -2)
            print("state loss", contact_model.loss[None] - form_loss)

            ## visualization
            viz_scale = 0.2
            viz_offset = [0.5, 0.2]

            if not Off_screen:
                contact_model.gripper.fem_sensor1.extract_markers(0)

                init_2d = contact_model.gripper.fem_sensor1.virtual_markers.to_numpy()
                marker_2d = contact_model.gripper.fem_sensor1.predict_markers.to_numpy()
                contact_model.draw_markers(init_2d, marker_2d, gui2)
                contact_model.draw_perspective(0)
                gui1.circles(viz_scale * contact_model.draw_pos.to_numpy() + viz_offset, radius=15, color=0x039dfc)
                gui1.circles(viz_scale * contact_model.draw_pos1.to_numpy() + viz_offset, radius=2, color=0xe6c949)
                gui1.circles(viz_scale * contact_model.draw_pos2.to_numpy() + viz_offset, radius=2, color=0xf542a1)
                contact_model.draw_triangles(contact_model.gripper.fem_sensor1, gui3, 0, 0, 90, viz_scale, viz_offset)

                gui1.show()
                gui2.show()
                gui3.show()

                camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.RMB)
                scene.set_camera(camera)
                scene.ambient_light((0.8, 0.8, 0.8))
                scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

                contact_model.draw_3d_scene(0)
                contact_model.scale_mesh_visualize()
                scene.particles(contact_model.draw_pos_3d, color = (0.68, 0.26, 0.19), radius = contact_model.p_rad/contact_model.view_scale)
                scene.particles(contact_model.draw_fem1_3d, per_vertex_color = contact_model.color_fem1_3d, radius = 0.008)
                scene.particles(contact_model.draw_fem2_3d, per_vertex_color = contact_model.color_fem2_3d, radius = 0.01)
                scene.mesh(contact_model.gripper.gripper_base.vis_ti_vertices, contact_model.gripper.gripper_base.vis_ti_faces, contact_model.gripper.gripper_base.vis_ti_normals)
                scene.mesh(contact_model.gripper.gripper_finger1.vis_ti_vertices, contact_model.gripper.gripper_finger1.vis_ti_faces, contact_model.gripper.gripper_finger1.vis_ti_normals)
                scene.mesh(contact_model.gripper.gripper_finger2.vis_ti_vertices, contact_model.gripper.gripper_finger2.vis_ti_faces, contact_model.gripper.gripper_finger2.vis_ti_normals)
                scene.lines(contact_model.draw_tableline, color = (0.28, 0.68, 0.99), width = 2.0)
                canvas.scene(scene)
                window.show()

        print("Begin backward")

        loss_frame = 0

        for ts in range(num_total_steps-2, -1, -1):
                print("BP", ts)
                contact_model.clear_all_grad()

                if USE_STATE:
                    contact_model.compute_rope_pos_loss(num_sub_steps-1)

                if USE_TACTILE:
                    contact_model.compute_contact_force(num_sub_steps - 2)
                    contact_model.compute_force_loss()
                    contact_model.compute_contact_loc(num_sub_steps - 2)
                    contact_model.compute_loc_loss()
                if USE_TACTILE:

                    contact_model.compute_loc_loss.grad()
                    contact_model.compute_contact_loc.grad(num_sub_steps - 2)

                    contact_model.compute_force_loss.grad()
                    contact_model.compute_contact_force.grad(num_sub_steps - 2)

                if USE_STATE:
                    contact_model.compute_rope_pos_loss.grad(num_sub_steps-1)

                # Loss grad

                for ss in range(num_sub_steps-2, -1, -1):
                    contact_model.update_grad(ss)

                contact_model.gripper.set_sensor_vel_grad(0)
                for ss in range(num_sub_steps-2, -1, -1):
                    contact_model.gripper.kinematic_grad(ss)
                contact_model.gripper.set_sensor_pos_grad(num_sub_steps-1)
                contact_model.set_pos_control.grad(ts)
                grad_p = contact_model.p_gripper.grad[ts]
                grad_o = contact_model.o_gripper.grad[ts]
                grad_w = contact_model.w_gripper.grad[ts]

                lr_p = 1e-2
                lr_o = 1e-2
                lr_w = 1e-2
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
                    for ss in range(num_sub_steps-1):
                        contact_model.gripper.kinematic(ss)
                    contact_model.gripper.set_sensor_vel(0)
                    contact_model.gripper.set_sensor_pos_bp()
                    contact_model.reset()
                    for ss in range(num_sub_steps-1):
                        contact_model.update(ss)


        losses.append(loss_frame)
        print("# Iter ", opts, "Opt step loss: ", loss_frame)


        if not os.path.exists(f"lr_cable_manipulation_{args.use_state}_tactile_{args.use_tactile}"):
            os.mkdir(f"lr_cable_manipulation_{args.use_state}_tactile_{args.use_tactile}")

        if not os.path.exists(f"results"):
            os.mkdir(f"results")

        if opts %5 == 0 or opts == num_opt_steps-1:
            plt.title("Trajectory Optimization")
            plt.ylabel("Loss")
            plt.xlabel("Iter") # "Gradient Descent Iterations"
            plt.plot(losses)
            plt.savefig(os.path.join(f"lr_cable_manipulation_{args.use_state}_tactile_{args.use_tactile}",f"cable_manipulation_{args.use_state}_tactile_{args.use_tactile}_{opts}.png"))
            np.save(os.path.join(f"lr_cable_manipulation_{args.use_state}_tactile_{args.use_tactile}", f"control_p_gripper_{opts}.npy"), contact_model.p_gripper.to_numpy())
            np.save(os.path.join(f"lr_cable_manipulation_{args.use_state}_tactile_{args.use_tactile}", f"control_o_gripper_{opts}.npy"), contact_model.o_gripper.to_numpy())
            np.save(os.path.join(f"lr_cable_manipulation_{args.use_state}_tactile_{args.use_tactile}", f"control_w_gripper_{opts}.npy"), contact_model.w_gripper.to_numpy())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_state", action = "store_true", help = "whether to use state loss")
    parser.add_argument("--use_tactile", action = "store_true", help = "whether to use tactile loss")

    args = parser.parse_args()
    USE_STATE = args.use_state
    USE_TACTILE = args.use_tactile
    main()
