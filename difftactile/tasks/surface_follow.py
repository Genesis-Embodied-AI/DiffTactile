"""
a surface following task
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

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
import trimesh

from difftactile.sensor_model.fem_sensor import FEMDomeSensor
from difftactile.object_model.rigid_static import RigidObj


TI_TYPE = ti.f32
NP_TYPE = np.float32



@ti.data_oriented
class Contact:
    def __init__(self, use_tactile, use_state, dt=5e-5, total_steps=300, sub_steps = 80, obj=None):
        self.iter = 0
        
        self.dt = dt
        self.total_steps = total_steps
        self.press_steps = 15
        self.decontact_ts = self.press_steps # when the sensor detaches from the surface
        self.decontact_flag = False
        self.sub_steps = sub_steps
        self.dim = 3
        self.fem_sensor1 = FEMDomeSensor(dt, sub_steps)
        self.space_scale = 10.0
        self.obj_scale = 8.0
        self.use_tactile = use_tactile
        self.use_state = use_state
        self.mpm_object = RigidObj(dt=dt, 
                                    sub_steps=sub_steps, 
                                    obj_name=obj,
                                    space_scale = self.space_scale,
                                    obj_scale = self.obj_scale,
                                    density = 1.5,
                                    rho = 4.0)
        self.num_sensor = 1
        self.target_sensor_pos = ti.Vector.field(self.dim, dtype = ti.f32, shape = ())

        self.init()

        self.view_phi = 0
        self.view_theta = 0
        
        self.kn = ti.field(dtype=float, shape=(), needs_grad=True)
        self.kd = ti.field(dtype=float, shape=(), needs_grad=True)
        self.kt = ti.field(dtype=float, shape=(), needs_grad=True)
        self.friction_coeff = ti.field(dtype=float, shape=(), needs_grad=True)#0.5

        self.kn[None] = 34.53
        self.kd[None] = 269.44
        self.kt[None] = 154.78
        self.friction_coeff[None] = 43.85
        self.fem_sensor1.mu[None] = 1294.01
        self.fem_sensor1.lam[None] = 9201.11
        self.contact_idx = ti.Vector.field(self.num_sensor, dtype=int, shape=(self.sub_steps, self.mpm_object.n_particles)) # indicating the contact seg idx of fem sensor for closested triangle mesh
        self.total_ext_f = ti.Vector.field(3, dtype=float, shape=())

        # control parameters
        self.dim = 3
        ### position control
        self.p_sensor1 = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.total_steps), needs_grad=True)
        self.o_sensor1 = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.total_steps), needs_grad=True)

        # for grad backward
        self.loss = ti.field(float, (), needs_grad=True)
        # # here we define the contact force as the target and then optimize the contact parameters
        self.target_force = ti.Vector.field(self.dim, float, shape=())
        self.target_loc = ti.Vector.field(self.dim, float, shape=())

        self.predict_force = ti.Vector.field(self.dim, float, (), needs_grad=True)
        self.predict_loc = ti.Vector.field(self.dim, float, (), needs_grad=True)
        self.contact_force = ti.Vector.field(self.dim, float, (), needs_grad=True)
        self.contact_loc = ti.Vector.field(self.dim, float, (), needs_grad=True)

        # visualization
        self.surf_offset1 = ti.Vector.field(2, float, self.fem_sensor1.num_triangles)
        self.surf_init_pos1 = ti.Vector.field(2, float, self.fem_sensor1.num_triangles)
        self.press_offset1 = ti.field(float, self.fem_sensor1.num_triangles)

        self.draw_pos2 = ti.Vector.field(2, float, self.fem_sensor1.n_verts) # elastomer1's pos
        self.draw_pos3 = ti.Vector.field(2, float, self.mpm_object.n_particles) # object's particle

        self.alpha = ti.field(float, ())
        self.beta = ti.field(float, ())
        self.alpha[None] = 100.0
        self.beta[None] = 1.0

    def init(self):
        self.ball_pos = [0.0, 0.0, 0.0]
        self.ball_ori = [-90.0, 0.0, 0.0]
        self.ball_vel = [0.0, 0.0, 0.0]

        self.mpm_object.init(self.ball_pos, self.ball_ori, self.ball_vel)

        ### extract the height of the obj
        obj_pos = self.mpm_object.x_0.to_numpy()[0,:]
        sensor_pos = self.fem_sensor1.init_x.to_numpy()
        self.init_h = np.max(sensor_pos[:,1]) + np.max(obj_pos[:,1]) 

        ## testing
        rx1 = 0.0
        ry1 = 0.0
        rz1 = -180.0
        t_dx1 = -1.50
        t_dy1 = self.init_h  #+ 0.5 #- 1.50 #+ (0.5 - 0.328)
        t_dz1 = 0.0

        self.fem_sensor1.init(rx1, ry1, rz1, t_dx1, t_dy1, t_dz1)

        self.in_contact = False
        self.contact_timestamp = 0


    @ti.kernel
    def init_pos_control(self):
        # # only pressing to initial the contact
        vx1 = 0.00; vy1 = 0.075/1.0; vz1 = 0.0

        for i in range(0, self.press_steps):
            self.p_sensor1[i] = ti.Vector([vx1, vy1, vz1])
            self.o_sensor1[i] = ti.Vector([0.0, 0.0, 0.0])

        vx1 = -0.1/1.0; vy1 = 0.0; vz1 = 0.0
        rx1 = 0.0; ry1 = 0.0; rz1 = 0.0

        for i in range(self.press_steps, self.total_steps):
            self.p_sensor1[i] = ti.Vector([vx1, vy1, vz1])
            self.o_sensor1[i] = ti.Vector([rx1, ry1, rz1])

    def load_pos_control(self):
        cur_pos = np.load(os.path.join(f"lr_surface_follow_{args.use_state}_tactile_{args.use_tactile}", f"control_p_gripper_15.npy"))
        cur_ori = np.load(os.path.join(f"lr_surface_follow_{args.use_state}_tactile_{args.use_tactile}", f"control_p_gripper_15.npy"))
        

        for i in range(len(cur_pos)):
            if np.isnan(cur_pos[i]).any() or np.isnan(cur_ori[i]).any() :
                cur_pos[i] = cur_pos[i-1]
                cur_ori[i] = cur_ori[i-1]
        self.p_sensor1.from_numpy(cur_pos)
        self.o_sensor1.from_numpy(cur_ori)


        
    @ti.kernel
    def set_pos_control(self, f:ti.i32):
      
        self.fem_sensor1.d_pos[None] = self.p_sensor1[f]
        self.fem_sensor1.d_ori[None] = self.o_sensor1[f]

    def update(self, f):
        self.fem_sensor1.update(f)
        self.check_collision(f)
        self.collision(f)
        self.mpm_object.update(f)
        self.fem_sensor1.update2(f)
        
    def update_grad(self, f):

        self.fem_sensor1.update2.grad(f)
        self.mpm_object.update.grad(f)
        self.collision.grad(f)
        self.clamp_grad(f)
        self.fem_sensor1.update.grad(f)

    @ti.kernel
    def clamp_grad(self, f:ti.i32):
        for i in range(self.fem_sensor1.n_verts):
            self.fem_sensor1.pos.grad[f, i] = ti.math.clamp(self.fem_sensor1.pos.grad[f, i], -1000.0, 1000.0)
            self.fem_sensor1.vel.grad[f, i] = ti.math.clamp(self.fem_sensor1.vel.grad[f, i], -1000.0, 1000.0)
       

    @ti.kernel
    def clear_loss_grad(self):
        self.kn.grad[None] = 0.0
        self.kd.grad[None] = 0.0
        self.kt.grad[None] = 0.0
        self.friction_coeff.grad[None] = 0.0
        self.contact_loc.grad[None].fill(0.0)
        self.contact_force.grad[None].fill(0.0)
        self.predict_loc.grad[None].fill(0.0)
        self.predict_force.grad[None].fill(0.0)
        self.loss[None] = 0.0
        self.loss.grad[None] = 1.0
        self.p_sensor1.grad.fill(0.0)
        self.o_sensor1.grad.fill(0.0)

    def clear_traj_grad(self):
        self.fem_sensor1.clear_loss_grad()
        self.mpm_object.clear_loss_grad()
        self.clear_loss_grad()

    def clear_all_grad(self):
        self.clear_traj_grad()
        self.fem_sensor1.clear_step_grad(self.sub_steps)
        self.mpm_object.clear_step_grad(self.sub_steps)
    
    def clear_step_grad(self):
        self.clear_traj_grad()
        self.fem_sensor1.clear_step_grad(self.sub_steps -1 )
        self.mpm_object.clear_step_grad(self.sub_steps -1 )


    def reset(self):
        self.fem_sensor1.reset_contact()
        self.mpm_object.reset()
        self.contact_idx.fill(-1)
        self.predict_force[None].fill(0.0)
        self.predict_loc[None].fill(0.0)
        self.contact_force[None].fill(0.0)
        self.contact_loc[None].fill(0.0)

    def load_target(self):
        self.target_force[None] = ti.Vector([300.0, -300.0, 0.0]) # only use normal force
        self.target_loc[None] = ti.Vector([0.0, 0.0, 0.0]) # don't use the height
        self.target_sensor_pos[None] = ti.Vector([-0.036, 3.308, 0.0])

    @ti.kernel
    def compute_contact_force(self, f:ti.i32):
        for i in range(self.fem_sensor1.num_triangles):
            a, b, c = self.fem_sensor1.contact_seg[i]
            self.contact_force[None] += 1/6 * self.fem_sensor1.external_force_field[f,a]
            self.contact_force[None] += 1/6 * self.fem_sensor1.external_force_field[f,b]
            self.contact_force[None] += 1/6 * self.fem_sensor1.external_force_field[f,c]

    @ti.kernel
    def compute_contact_loc(self, f:ti.i32):
        ### CAVEAT: if there's no contact, the grad might be nan!!
        for i in range(self.fem_sensor1.num_triangles):
            a, b, c = self.fem_sensor1.contact_seg[i]
            v_pa = self.fem_sensor1.inv_rot[None] @ self.virtual_pos[f, a]
            v_pb = self.fem_sensor1.inv_rot[None] @ self.virtual_pos[f, b]
            v_pc = self.fem_sensor1.inv_rot[None] @ self.virtual_pos[f, c]

            init_pa = self.fem_sensor1.inv_rot[None] @ self.pos[f, a]
            init_pb = self.fem_sensor1.inv_rot[None] @ self.pos[f, b]
            init_pc = self.fem_sensor1.inv_rot[None] @ self.pos[f, c]
            # weight the loc based on pressing distance
            self.contact_loc[None] += 1/3 * init_pa * (v_pa[1]-init_pa[1])**2
            self.contact_loc[None] += 1/3 * init_pb * (v_pb[1]-init_pb[1])**2 
            self.contact_loc[None] += 1/3 * init_pc * (v_pc[1]-init_pc[1])**2 

    @ti.kernel
    def compute_force_loss(self):
        self.predict_force[None] = self.fem_sensor1.inv_rot[None] @ self.contact_force[None]
        self.loss[None] += self.beta[None]*(self.predict_force[None][0] - self.target_force[None][0])**2 + (self.predict_force[None][1] - self.target_force[None][1])**2 + (self.predict_force[None][2] - self.target_force[None][2])**2

    @ti.kernel
    def compute_loc_loss(self):
        self.predict_loc[None] = self.fem_sensor1.inv_rot[None] @ self.contact_loc[None]
        self.loss[None] += (self.predict_loc[None][0] - self.target_loc[None][0])**2 + (self.predict_loc[None][2] - self.target_loc[None][2])**2

    @ti.kernel
    def compute_sensor_dis_loss(self):

        self.loss[None] += self.alpha[None] *( (self.fem_sensor1.trans_h[None][0,3] - self.target_sensor_pos[None][0])**2 + (self.fem_sensor1.trans_h[None][1,3] - self.target_sensor_pos[None][1])**2 +\
                                             (self.fem_sensor1.trans_h[None][2,3] - self.target_sensor_pos[None][2])**2 )


    @ti.func
    def calculate_contact_force(self, sdf, norm_v, relative_v):
        shear_factor_p0 = ti.Vector([0.0, 0.0, 0.0])
        shear_vel_p0 = ti.Vector([0.0, 0.0, 0.0])

        relative_vel_p0 = relative_v
        normal_vel_p0 = ti.max(norm_v.dot(relative_vel_p0), 0)
        normal_factor_p0 = -(self.kn[None] + self.kd[None] * normal_vel_p0)* sdf * norm_v

        ### also add spring
        shear_vel_p0 = relative_vel_p0 - norm_v.dot(relative_vel_p0) * norm_v
        shear_vel_norm_p0 = shear_vel_p0.norm()
        if shear_vel_norm_p0 > 1e-4:
            shear_factor_p0 = 1.0*(shear_vel_p0/shear_vel_norm_p0) * ti.min(self.kt[None] * shear_vel_norm_p0, self.friction_coeff[None]*normal_factor_p0.norm())
        ext_v = normal_factor_p0 + shear_factor_p0
        # print(normal_factor_p0, shear_factor_p0)
        return ext_v, normal_factor_p0, shear_factor_p0

    @ti.kernel
    def check_collision(self, f:ti.i32):
        for p in range(self.mpm_object.n_particles):
            cur_p = self.mpm_object.x_0[f, p]
            min_idx1 = self.fem_sensor1.find_closest(cur_p, f)
            self.contact_idx[f, p] = [min_idx1]

    @ti.kernel
    def collision(self, f:ti.i32):
        for p in range(self.mpm_object.n_particles):
            ### boundary condition with sensor elastomer sdf
            # use the center of the grid to check the sdf
            cur_p = self.mpm_object.x_0[f, p]
            cur_v = self.mpm_object.v_0[f, p]
            min_idx1 = self.contact_idx[f, p]
            cur_sdf1, cur_norm_v1, cur_relative_v1, contact_flag1 = self.fem_sensor1.find_sdf(cur_p, cur_v, min_idx1, f)
            if contact_flag1:
                ext_v1, ext_n1, ext_t1 = self.calculate_contact_force(cur_sdf1, -1*cur_norm_v1, -1*cur_relative_v1)
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
            gui.arrows(draw_points, 1.0*offset, radius=2, color=0xe6c949)
        


    def draw_markers_cv(self, tracked_markers, init_markers, showScale):
        img = np.zeros((480, 640, 3))
        markerCenter=np.around(init_markers[:,0:2]).astype(np.int16)
        for i in range(init_markers.shape[0]):
            marker_motion = tracked_markers[i] - init_markers[i]
            cv2.arrowedLine(img,(markerCenter[i,0], markerCenter[i,1]), \
                (int(init_markers[i,0]+marker_motion[0]*showScale), int(init_markers[i,1]+marker_motion[1]*showScale)),\
                (0, 255, 255),2)
        return img

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


   
    def apply_action(self, action, ts):
        if ts < self.press_steps:
            d_pos = np.array([action[0], action[1] + 0.075, action[2]])
            d_ori = np.array([action[3], action[4], action[5]])
        else:
            d_pos = np.array([action[0] - 0.1, action[1], action[2]])
            d_ori = np.array([action[3], action[4], action[5]])
        
        self.fem_sensor1.d_pos.from_numpy(d_pos)
        self.fem_sensor1.d_ori.from_numpy(d_ori)
        self.fem_sensor1.set_pose_control()
        self.fem_sensor1.set_control_vel(0)
        self.fem_sensor1.set_vel(0)
        self.reset()
        for ss in range(self.sub_steps - 1):
            self.update(ss)
        self.memory_to_cache(ts)
        
    def prepare_env(self):
        self.init_pos_control()
        self.load_target()
        
    def calculate_force(self, ts):
        if self.use_tactile:
            self.compute_contact_force(self.sub_steps - 2)
    
    def compute_loss(self, ts):
        if self.use_tactile:
            self.compute_force_loss()
        if self.use_state:
            self.compute_sensor_dis_loss()
        
    def render(self, gui1, gui2, gui3):
        viz_scale = 0.1
        viz_offset = [0.5, 0.2]
        self.fem_sensor1.extract_markers(0)
        init_2d = self.fem_sensor1.virtual_markers.to_numpy()
        marker_2d = self.fem_sensor1.predict_markers.to_numpy()
        self.draw_markers(init_2d, marker_2d, gui2)
        self.draw_perspective(0)
        gui1.circles(viz_scale * self.draw_pos3.to_numpy() + viz_offset, radius=2, color=0x039dfc)
        gui1.circles(viz_scale * self.draw_pos2.to_numpy() + viz_offset, radius=2, color=0xe6c949)
        self.draw_triangles(self.fem_sensor1, gui3, 0, 0, 90, viz_scale, viz_offset)
        
        gui1.show()
        gui2.show()
        gui3.show()

def transform_2d(point, angle, translate):
    theta = np.radians(angle)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    new_point = np.matmul(rot_mat, point.T).T + translate
    return new_point

def main():
    ti.init(arch=ti.gpu, device_memory_GB=4)
    
    obj_name = "Random-surface.stl"
    num_sub_steps = 50
    num_total_steps = 1000
    num_opt_steps = 200
    dt = 5e-4
    contact_model = Contact(use_tactile=USE_TACTILE, use_state=USE_STATE, dt=dt, total_steps = num_total_steps, sub_steps = num_sub_steps, obj=obj_name)

    if not off_screen:
        gui1 = ti.GUI("Contact Viz")
        gui2 = ti.GUI("Force Map 1")
        gui3 = ti.GUI("Deformation Map 1")

    losses = []
    contact_model.init_pos_control()
    contact_model.load_target()


    form_loss = 0

    for opts in range(num_opt_steps):
        print("Opt # step ======================", opts)
        contact_model.init()
        contact_model.clear_all_grad()
        
        for ts in range(num_total_steps-1):
            print("# FP Iter ", ts)
            contact_model.iter = ts
            contact_model.set_pos_control(ts)
            contact_model.fem_sensor1.set_pose_control()
            contact_model.fem_sensor1.set_control_vel(0)
            contact_model.fem_sensor1.set_vel(0)
            contact_model.reset()
            for ss in range(num_sub_steps - 1):
                contact_model.update(ss)
            contact_model.memory_to_cache(ts)
            contact_model.compute_contact_force(num_sub_steps - 2)
            form_loss = contact_model.loss[None]
            contact_model.compute_force_loss()
            form_loss = contact_model.loss[None]
            contact_model.compute_sensor_dis_loss()

            print("Control vec p/o: ", contact_model.p_sensor1[ts], contact_model.o_sensor1[ts])

            viz_scale = 0.1
            viz_offset = [0.5, 0.2]
            contact_model.fem_sensor1.extract_markers(0)
            init_2d = contact_model.fem_sensor1.virtual_markers.to_numpy()
            marker_2d = contact_model.fem_sensor1.predict_markers.to_numpy()
            if not off_screen:
                contact_model.draw_markers(init_2d, marker_2d, gui2)

            if not off_screen:
                contact_model.draw_perspective(0)
                gui1.circles(viz_scale * contact_model.draw_pos3.to_numpy() + viz_offset, radius=2, color=0x039dfc)
                gui1.circles(viz_scale * contact_model.draw_pos2.to_numpy() + viz_offset, radius=2, color=0xe6c949)
                contact_model.draw_triangles(contact_model.fem_sensor1, gui3, 0, 0, 90, viz_scale, viz_offset)

                gui1.show()
                gui2.show()
                gui3.show()


        ### backward!
        loss_frame = 0
        form_loss = 0

        for ts in range(num_total_steps-2, -1, -1):
            print("BP", ts)
            contact_model.clear_all_grad()

            if USE_TACTILE:
                contact_model.compute_contact_force(num_sub_steps - 2)
                form_loss = contact_model.loss[None]
                contact_model.compute_force_loss()
                print("force loss", contact_model.loss[None] - form_loss)

            if USE_STATE:
                form_loss = contact_model.loss[None]
                contact_model.compute_sensor_dis_loss()
                print("state loss", contact_model.loss[None] - form_loss)
                contact_model.compute_sensor_dis_loss.grad()

            if USE_TACTILE:
                contact_model.compute_force_loss.grad()
                contact_model.compute_contact_force.grad(num_sub_steps - 2)

            for ss in range(num_sub_steps-2, -1, -1):
                contact_model.update_grad(ss)

            contact_model.fem_sensor1.set_vel.grad(0)
            contact_model.fem_sensor1.set_control_vel.grad(0)
            contact_model.fem_sensor1.set_pose_control.grad()
            contact_model.set_pos_control.grad(ts)
            
            ### optimization with grad backpropagation        
            grad_p = contact_model.p_sensor1.grad[ts]
            grad_o = contact_model.o_sensor1.grad[ts]

            lr_p = 1e-5/20#5e-1#state#1e-5/20
            lr_o = 1e-3/20#5e-1#state#1e-3/20
            contact_model.p_sensor1[ts] -= lr_p * grad_p
            contact_model.o_sensor1[ts] -= lr_o * grad_o

            loss_frame += contact_model.loss[None]

            print("# BP Iter: ", ts, " loss: ", contact_model.loss[None])
            print("P/O grads: ", grad_p, grad_o)
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

        losses.append(loss_frame)
        print("# Iter ", opts, "Opt step loss: ", loss_frame)
        

        if not os.path.exists(f"lr_surface_follow_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}"):
            os.mkdir(f"lr_surface_follow_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}")

        if not os.path.exists(f"results"):
            os.mkdir(f"results")
        
        if opts %5 == 0 or opts == num_opt_steps-1:
            plt.title("Trajectory Optimization")
            plt.ylabel("Loss")
            plt.xlabel("Iter") # "Gradient Descent Iterations"
            plt.plot(losses)
            plt.savefig(os.path.join(f"lr_surface_follow_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}",f"surface_follow_{args.use_state}_tactile_{args.use_tactile}_{opts}.png"))
            np.save(os.path.join(f"lr_surface_follow_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}", f"control_p_gripper_{opts}.npy"), contact_model.p_sensor1.to_numpy())
            np.save(os.path.join(f"lr_surface_follow_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}", f"control_o_gripper_{opts}.npy"), contact_model.o_sensor1.to_numpy())
            np.save(os.path.join(f"lr_surface_follow_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}", f"losses_{opts}.npy"), np.array(losses))


        if loss_frame <= np.min(losses):
            best_p = contact_model.p_sensor1.to_numpy()
            best_o = contact_model.o_sensor1.to_numpy()
            np.save(os.path.join(f"lr_surface_follow_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}","control_pos_best.npy"), best_p)
            np.save(os.path.join(f"lr_surface_follow_state_{args.use_state}_tactile_{args.use_tactile}_{args.times}","control_ori_best.npy"), best_o)
            print("Best traj saved!")
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--use_state", action = "store_true", help = "whether to use state loss")
    parser.add_argument("--use_tactile", action = "store_true", help = "whether to use tactile loss")
    parser.add_argument("--times", type = int, default = 1)

    args = parser.parse_args()
    USE_STATE = args.use_state
    USE_TACTILE = args.use_tactile
    main()
