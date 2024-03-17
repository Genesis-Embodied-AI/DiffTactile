"""
load gripper mesh model and build kinematic chain
use centermeter as units
the space is 10 cm x 10 cm x 10 cm
"""

import os
import taichi as ti
import trimesh
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import math
import torch
from difftactile.sensor_model.gripper_fem import FEMDomeSensor

TI_TYPE = ti.f32
TC_TYPE = torch.float32
NP_TYPE = np.float32

class MeshLoader:
    def __init__(self, data_path, scale, vis_scale=10.0):
        self.data_path = data_path # ending with obj or stl
        self.raw_mesh = trimesh.load(self.data_path, force='mesh', skip_texture=True)
        origin = [0, 0, 0]
        ss = trimesh.transformations.scale_matrix(scale, origin)
        self.raw_mesh.apply_transform(ss)
        self.vis_mesh = trimesh.Trimesh()
        self.vis_scale = vis_scale

        self.ti_vertices = ti.Vector.field(3, dtype=ti.f32, shape=(len(self.raw_mesh.vertices)))
        self.ti_faces = ti.field(dtype=ti.i32, shape=(len(self.raw_mesh.faces)*3))
        self.ti_normals = ti.Vector.field(3, dtype=ti.f32, shape=(len(self.raw_mesh.vertex_normals)))

        self.vis_ti_vertices = ti.Vector.field(3, dtype=ti.f32, shape=(len(self.raw_mesh.vertices)))
        self.vis_ti_faces = ti.field(dtype=ti.i32, shape=(len(self.raw_mesh.faces)*3))
        self.vis_ti_normals = ti.Vector.field(3, dtype=ti.f32, shape=(len(self.raw_mesh.vertex_normals)))

    def update(self, trans_h):
        self.trans_h = trans_h
        self.raw_mesh.apply_transform(self.trans_h)
        self.vertices = np.array(self.raw_mesh.vertices)
        self.faces = np.array(self.raw_mesh.faces).flatten()
        self.normals = np.array(self.raw_mesh.vertex_normals)

        self.ti_vertices.from_numpy(self.vertices)
        self.ti_normals.from_numpy(self.normals)
        self.ti_faces.from_numpy(self.faces)

    def scale_visualize(self):
        self.vis_mesh = self.raw_mesh.copy()
        origin = [0, 0, 0]
        ss = trimesh.transformations.scale_matrix(self.vis_scale, origin)
        self.vis_mesh.apply_transform(ss)

        self.vis_vertices = np.array(self.vis_mesh.vertices)
        self.vis_faces = np.array(self.vis_mesh.faces).flatten()
        self.vis_normals = np.array(self.vis_mesh.vertex_normals)

        self.vis_ti_vertices.from_numpy(self.vis_vertices)
        self.vis_ti_normals.from_numpy(self.vis_normals)
        self.vis_ti_faces.from_numpy(self.vis_faces)

    def save_mesh(self, name):
        self.raw_mesh.export(name)

@ti.data_oriented
class Gripper:
    def __init__(self, data_path, dt=5e-5, total_steps=300, sub_steps=80, mesh_vis_scale = 1.0):
        self.data_path = data_path
        self.gripper_base_path = os.path.join(self.data_path, "gripper-base.stl")
        self.gripper_finger_path = os.path.join(self.data_path, "gripper-finger.stl")
        self.sub_steps = sub_steps

        self.base_height = 2.0
        self.base_width = 14.0
        self.finger_height = 7.5
        self.finger_width = 1.4

        self.mesh_vis_scale = mesh_vis_scale

        
        self.ee_delta = ti.Matrix.field(4,4, ti.f32, shape = (sub_steps), needs_grad=True)
        self.finger1_delta = ti.Matrix.field(4,4,ti.f32,shape=(sub_steps), needs_grad=True)
        self.finger2_delta = ti.Matrix.field(4,4,ti.f32,shape=(sub_steps), needs_grad=True)
        self.ee = ti.Matrix.field(4,4, ti.f32, shape = (sub_steps), needs_grad=True)
        self.finger1_ee = ti.Matrix.field(4,4, ti.f32, shape = (sub_steps), needs_grad=True) # right
        self.finger2_ee = ti.Matrix.field(4,4, ti.f32, shape = (sub_steps), needs_grad=True) # left
        self.finger1_local = ti.Matrix.field(4, 4, ti.f32, shape=(sub_steps), needs_grad=True)
        self.finger2_local = ti.Matrix.field(4, 4, ti.f32, shape=(sub_steps), needs_grad=True)

        self.ee_local = ti.Matrix.field(4,4, ti.f32, shape = (sub_steps), needs_grad=True) ## ee's pose in local
        self.finger1_t = ti.Matrix.field(4,4, ti.f32, shape = ()) # right
        self.finger2_t = ti.Matrix.field(4,4, ti.f32, shape = ()) # left
        self.ee_world = ti.Matrix.field(4,4, ti.f32, shape = ()) ## ee to world coord

        self.dt = dt
        self.total_steps = total_steps
        self.sub_steps = sub_steps
        self.dim = 3
        self.fem_sensor1 = FEMDomeSensor(dt, sub_steps)
        self.fem_sensor2 = FEMDomeSensor(dt, sub_steps)
        self.num_sensor = 2

        # Control parameters
        self.d_pos = ti.Vector.field(3, ti.f32, shape = (), needs_grad=True)
        self.d_ori = ti.Vector.field(3, ti.f32, shape = (), needs_grad=True)
        self.d_gripper = ti.field(dtype=ti.f32, shape = (), needs_grad=True)
        self.grip_delta = ti.field(dtype=ti.f32, shape = (sub_steps), needs_grad=True)

        self.cache = dict()

   

    def init_mesh(self):
        self.gripper_base = MeshLoader(self.gripper_base_path, 1, 1/self.mesh_vis_scale)
        self.gripper_finger1 = MeshLoader(self.gripper_finger_path, 1, 1/self.mesh_vis_scale)
        self.gripper_finger2 = MeshLoader(self.gripper_finger_path, 1, 1/self.mesh_vis_scale)


    @ti.kernel
    def init_params(self, rot:ti.types.vector(3, ti.f32), trans:ti.types.vector(3, ti.f32), grip_width:ti.f32):
        
        # init kinematic
        ee_trans, _ = self.eul2mat(rot, trans)
        self.ee_world[None] = ee_trans
        self.ee_local[0] = ti.Matrix.identity(dt=ti.f32, n=4)
        self.ee[0] = self.ee_world[None]
        self.ee_delta[0] = self.ee[0] @ self.ee_local[0]
        self.finger1_local[0] = ti.Matrix.identity(dt=ti.f32, n=4)
        self.finger2_local[0] = ti.Matrix.identity(dt=ti.f32, n=4)
        # gripper ee to finfer1 ee
        e2f1_rot = ti.Vector([-90, 0, -90])
        viz1_rot = ti.Vector([0, 0, -90])

        e2f1_t = ti.Vector([-self.base_width/2 + grip_width, -self.base_height - self.finger_height, 0])
        e2f1_trans, _ = self.eul2mat(e2f1_rot, e2f1_t)
        viz1_t = ti.Vector([-self.base_width/2 + grip_width - 1.5, -self.base_height - self.finger_height, 0])
        viz1_trans, _ = self.eul2mat(viz1_rot, viz1_t)

        self.finger1_t[None] = e2f1_trans
        self.finger1_ee[0] = self.ee[0] @ self.finger1_t[None] @ self.finger1_local[0]
        self.finger1_delta[0] = self.ee[0] @ viz1_trans @ self.finger1_local[0]

        # gripper ee to finfer2 ee
        e2f2_rot = ti.Vector([-90, 0, 90])
        viz2_rot = ti.Vector([0, 180, -90])

        e2f2_t = ti.Vector([self.base_width/2 - grip_width, -self.base_height - self.finger_height, 0])
        e2f2_trans, _ = self.eul2mat(e2f2_rot, e2f2_t)
        viz2_t = ti.Vector([self.base_width/2 - grip_width + 1.5, -self.base_height - self.finger_height, 0])
        viz2_trans, _ = self.eul2mat(viz2_rot, viz2_t)

        self.finger2_t[None] = e2f2_trans
        self.finger2_ee[0] = self.ee[0] @ self.finger2_t[None] @ self.finger2_local[0]
        self.finger2_delta[0] = self.ee[0] @ viz2_trans @ self.finger2_local[0]


    def init(self, rot:ti.types.vector(3, ti.f32), trans:ti.types.vector(3, ti.f32), grip_width:ti.f32):
        self.init_mesh()
        self.init_params(rot, trans, grip_width)

    def init_fem_sensor(self):
        matrix1 = self.finger1_ee[0].to_numpy()
        matrix2 = self.finger2_ee[0].to_numpy()
        self.fem_sensor1.init(matrix1)
        self.fem_sensor2.init(matrix2)

    @ti.kernel
    def grip_kinematic(self, f: ti.i32):
        d_grip = self.grip_delta[f]/2
        f_rot = ti.Vector([0, 0, 0])
        f1_t = ti.Vector([0, d_grip, 0])
        f1_delta, _ = self.eul2mat(f_rot, f1_t)
        f2_t = ti.Vector([0, d_grip, 0])
        f2_delta, _ = self.eul2mat(f_rot, f2_t)
        self.finger1_local[f+1] = f1_delta @ self.finger1_local[f]
        self.finger2_local[f+1] = f2_delta @ self.finger2_local[f]

        self.finger1_ee[f+1] = self.ee[f+1] @ self.finger1_t[None] @ self.finger1_local[f+1]
        self.finger2_ee[f+1] = self.ee[f+1] @ self.finger2_t[None] @ self.finger2_local[f+1]
        self.finger1_delta[f+1] = self.finger1_ee[f+1] @ self.finger1_ee[f].inverse()
        self.finger2_delta[f+1] = self.finger2_ee[f+1] @ self.finger2_ee[f].inverse()


    @ti.kernel
    def pos_transformation(self, f:ti.i32):
        ee_delta, _ = self.eul2mat(self.d_ori[None] * self.dt, self.d_pos[None]*self.dt)
        self.ee_delta[f+1] = self.ee_world[None] @ ee_delta @ (self.ee_world[None].inverse())
        self.ee_local[f+1] = ee_delta @ self.ee_local[f]
        self.ee[f+1] = self.ee_world[None] @ self.ee_local[f+1]
        self.grip_delta[f] = 2 *self.d_gripper[None] * self.dt

    @ti.kernel
    def fem_sensor1_set_control_vel(self, f:ti.i32):
        for i in range(self.fem_sensor1.n_verts):

            init_x = ti.Vector([self.fem_sensor1.init_x[i][0], self.fem_sensor1.init_x[i][1], self.fem_sensor1.init_x[i][2], 1.0])
            init_t_pos = self.finger1_ee[f] @ init_x
            after_t_pos =  self.finger1_ee[f+1] @ init_x

            self.fem_sensor1.control_vel[f, i][0] = (after_t_pos[0] - init_t_pos[0]) / self.fem_sensor1.dt
            self.fem_sensor1.control_vel[f, i][1] = (after_t_pos[1] - init_t_pos[1]) /self.fem_sensor1.dt
            self.fem_sensor1.control_vel[f, i][2] = (after_t_pos[2] - init_t_pos[2]) /self.fem_sensor1.dt

    @ti.kernel
    def fem_sensor2_set_control_vel(self, f:ti.i32):
        for i in range(self.fem_sensor2.n_verts):

            init_x = ti.Vector([self.fem_sensor2.init_x[i][0], self.fem_sensor2.init_x[i][1], self.fem_sensor2.init_x[i][2], 1.0])
            init_t_pos = self.finger2_ee[f] @ init_x
            after_t_pos =  self.finger2_ee[f+1] @ init_x

            self.fem_sensor2.control_vel[f, i][0] = (after_t_pos[0] - init_t_pos[0]) / self.fem_sensor2.dt
            self.fem_sensor2.control_vel[f, i][1] = (after_t_pos[1] - init_t_pos[1]) /self.fem_sensor2.dt
            self.fem_sensor2.control_vel[f, i][2] = (after_t_pos[2] - init_t_pos[2]) /self.fem_sensor2.dt


    def kinematic(self, f:ti.i32):
        self.pos_transformation(f)
        self.grip_kinematic(f)
        self.fem_sensor1_set_control_vel(f)
        self.fem_sensor2_set_control_vel(f)

    def kinematic_grad(self, f:ti.i32):
        self.fem_sensor1_set_control_vel.grad(f)
        self.fem_sensor2_set_control_vel.grad(f)
        self.grip_kinematic.grad(f)
        self.pos_transformation.grad(f)

    def set_sensor_vel(self, f:ti.i32):
        self.fem_sensor1.set_vel(f)
        self.fem_sensor2.set_vel(f)

    def set_sensor_vel_grad(self, f:ti.i32):
        self.fem_sensor1.set_vel.grad(f)
        self.fem_sensor2.set_vel.grad(f)

    @ti.kernel
    def fem_sensor1_set_rot(self, f: ti.i32):
        for i in range(3):
            for j in range(3):
                self.fem_sensor1.rot_h[None][i, j] = self.finger1_ee[f][i, j]

    @ti.kernel
    def fem_sensor2_set_rot(self, f: ti.i32):
        for i in range(3):
            for j in range(3):
                self.fem_sensor2.rot_h[None][i, j] = self.finger2_ee[f][i, j]

    @ti.kernel
    def fem_sensor1_set_pos_control(self, f:ti.i32):
        self.fem_sensor1.trans_h[None] = self.finger1_ee[f]
        self.fem_sensor1.inv_trans_h[None] = self.fem_sensor1.trans_h[None].inverse()
        self.fem_sensor1.inv_rot[None] = self.fem_sensor1.rot_h[None].inverse()

    @ti.kernel
    def fem_sensor2_set_pos_control(self, f:ti.i32):
        self.fem_sensor2.trans_h[None] = self.finger2_ee[f]
        self.fem_sensor2.inv_trans_h[None] = self.fem_sensor2.trans_h[None].inverse()
        self.fem_sensor2.inv_rot[None] = self.fem_sensor2.rot_h[None].inverse()

    def set_sensor_pos(self, f:ti.i32):
        self.fem_sensor1_set_rot(f)
        self.fem_sensor2_set_rot(f)
        self.fem_sensor1_set_pos_control(f)
        self.fem_sensor2_set_pos_control(f)

    def set_sensor_pos_bp(self):
        self.fem_sensor1.set_pos_control_bp()
        self.fem_sensor2.set_pos_control_bp()


    def set_sensor_pos_grad(self, f:ti.i32):

        self.fem_sensor1_set_pos_control.grad(f)
        self.fem_sensor2_set_pos_control.grad(f)
        self.fem_sensor1_set_rot.grad(f)
        self.fem_sensor2_set_rot.grad(f)

    @ti.func
    def eul2mat(self, rot_v, trans_v):
        # rot_v: euler angles (degrees) for rotation (x,y,z)
        # trans_v: translation (x,y,z)
        rot_v_r = ti.math.radians(rot_v)
        rot_x = rot_v_r[0]
        rot_y = rot_v_r[1]
        rot_z = rot_v_r[2]
        mat_x = ti.Matrix([[1.0, 0.0, 0.0],[0.0, ti.cos(rot_x), -ti.sin(rot_x)],[0.0, ti.sin(rot_x), ti.cos(rot_x)]])
        mat_y = ti.Matrix([[ti.cos(rot_y), 0.0, ti.sin(rot_y)],[0.0, 1.0, 0.0],[-ti.sin(rot_y), 0.0, ti.cos(rot_y)]])
        mat_z = ti.Matrix([[ti.cos(rot_z), -ti.sin(rot_z), 0.0],[ti.sin(rot_z), ti.cos(rot_z), 0.0],[0.0, 0.0, 1.0]])
        mat_R = mat_x @ mat_y @ mat_z
        trans_h = ti.Matrix.identity(float, 4)
        trans_h[0:3, 0:3] = mat_R
        trans_h[0:3, 3] = trans_v
        return trans_h, mat_R

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        self.ee_local[target] = self.ee_local[source]
        self.finger1_ee[target] = self.finger1_ee[source]
        self.finger2_ee[target] = self.finger2_ee[source]
        self.finger1_local[target] = self.finger1_local[source]
        self.finger2_local[target] = self.finger2_local[source]

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        self.ee_local.grad[target] = self.ee_local.grad[source]
        self.finger1_ee.grad[target] = self.finger1_ee.grad[source]
        self.finger2_ee.grad[target] = self.finger2_ee.grad[source]
        self.finger1_local.grad[target] = self.finger1_local.grad[source]
        self.finger2_local.grad[target] = self.finger2_local.grad[source]

    @ti.kernel
    def add_step_to_cache(self, f:ti.i32,  cache_ee_local: ti.types.ndarray(),cache_finger1_ee: ti.types.ndarray(), cache_finger2_ee: ti.types.ndarray(), cache_finger1_local: ti.types.ndarray(), cache_finger2_local: ti.types.ndarray()):
        for j in range(4):
            for k in range(4):
                cache_ee_local[j,k] = self.ee_local[f][j,k]
        for j in range(4):
            for k in range(4):
                cache_finger1_ee[j,k] = self.finger1_ee[f][j,k]
        for j in range(4):
            for k in range(4):
                cache_finger2_ee[j,k] = self.finger2_ee[f][j,k]
        for j in range(4):
            for k in range(4):
                cache_finger1_local[j,k] = self.finger1_local[f][j,k]
        for j in range(4):
            for k in range(4):
                cache_finger2_local[j,k] = self.finger2_local[f][j,k]

    @ti.kernel
    def load_step_from_cache(self, f:ti.i32, cache_ee_local: ti.types.ndarray(),cache_finger1_ee: ti.types.ndarray(), cache_finger2_ee: ti.types.ndarray(), cache_finger1_local: ti.types.ndarray(), cache_finger2_local: ti.types.ndarray()):
        for j in range(4):
            for k in range(4):
                self.ee_local[f][j,k] = cache_ee_local[j,k]
        for j in range(4):
            for k in range(4):
                self.finger1_ee[f][j,k] = cache_finger1_ee[j,k]
        for j in range(4):
            for k in range(4):
                self.finger2_ee[f][j,k] = cache_finger2_ee[j,k]
        for j in range(4):
            for k in range(4):
                self.finger1_local[f][j,k] = cache_finger1_local[j,k]
        for j in range(4):
            for k in range(4):
                self.finger2_local[f][j,k] = cache_finger2_local[j,k]

    def memory_to_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.cache[cur_step_name] = dict()
        self.cache[cur_step_name]['ee_local'] = torch.zeros((4,4), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['finger1_ee'] = torch.zeros((4,4), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['finger2_ee'] = torch.zeros((4,4), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['finger1_local'] = torch.zeros((4,4), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['finger2_local'] = torch.zeros((4,4), dtype=TC_TYPE, device=device)
        self.add_step_to_cache(0, self.cache[cur_step_name]['ee_local'], self.cache[cur_step_name]['finger1_ee'], self.cache[cur_step_name]['finger2_ee'], self.cache[cur_step_name]['finger1_local'], self.cache[cur_step_name]['finger2_local'])
        self.copy_frame(self.sub_steps-1, 0)

    def memory_from_cache(self, t):
        cur_step_name = f'{t:06d}'
        self.copy_frame(0, self.sub_steps-1)
        self.copy_grad(0, self.sub_steps-1)
        self.clear_step_grad(self.sub_steps-1)
        self.load_step_from_cache(0, self.cache[cur_step_name]['ee_local'], self.cache[cur_step_name]['finger1_ee'], self.cache[cur_step_name]['finger2_ee'], self.cache[cur_step_name]['finger1_local'], self.cache[cur_step_name]['finger2_local'])

    @ti.kernel
    def clear_step_grad(self, f:ti.i32):
        for t in range(f):
            self.ee_delta.grad[t].fill(0.0)
            self.finger1_delta.grad[t].fill(0.0)
            self.finger2_delta.grad[t].fill(0.0)
            self.ee.grad[t].fill(0.0)
            self.finger1_ee.grad[t].fill(0.0)
            self.finger2_ee.grad[t].fill(0.0)
            self.finger1_local.grad[t].fill(0.0)
            self.finger2_local.grad[t].fill(0.0)
            self.ee_local.grad[t].fill(0.0)
            self.grip_delta.grad[t] = 0.0

    @ti.kernel
    def clear_loss_grad(self):

        self.d_pos.grad[None].fill(0.0)
        self.d_ori.grad[None].fill(0.0)
        self.d_gripper.grad[None] = 0.0
