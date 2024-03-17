"""
a class to describe rigid objects with mpm
"""
import os
import taichi as ti
from math import pi
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from difftactile.object_model.obj_loader import ObjLoader

TI_TYPE = ti.f32
TC_TYPE = torch.float32
NP_TYPE = np.float32

@ti.data_oriented
class RigidObj:
    def __init__(self, dt=5e-5, sub_steps=80, obj_name=None, space_scale = 1.0, obj_scale = 1.0, density = 2, rho = 1):
        self.sub_steps = sub_steps
        self.dt = dt
        self.init_pos = ti.Vector.field(3,dtype=ti.f32, shape=())
        self.init_vel = ti.Vector.field(3,dtype=ti.f32, shape=())
        self.rot_h = ti.Matrix.field(3, 3, ti.f32, shape = ())
        self.t_h = ti.Vector.field(3, ti.f32, shape = ())

        self.drot_h = ti.Matrix.field(3, 3, ti.f32, shape = ())
        self.dt_h = ti.Vector.field(3, ti.f32, shape = ())

        self.dim = 3
        self.n_grid = 64
        self.space_scale = space_scale
        self.obj_scale = obj_scale
        self.rho = rho * self.obj_scale
        self.particle_density = self.n_grid * density * obj_scale / space_scale
        self.gravity = ti.Vector([0.0, -9.8, 0.0])

        ## parameters for object
        self.obj_name = obj_name
        if self.obj_name is not None:
            data_path = os.path.join("..", "meshes", "objects", self.obj_name)
            obj_loader = ObjLoader(data_path, particle_density = int(self.particle_density))
            self.n_particles = 30 ** 3#30 ** 3
            obj_loader.generate_surface_particles(self.n_particles)
            self.particles = ti.Vector.field(3, dtype=float, shape=self.n_particles)
            self.particles.from_numpy((obj_loader.particles * self.obj_scale).astype(np.float32))
            print("Object model is loaded!")
        else:
            self.n_particles = 20 ** 3


        self.dx_0 = float(self.space_scale / self.n_grid)
        self.inv_dx_0 =  1 / self.dx_0
        self.p_vol, self.p_rho =  (self.dx_0 * self.obj_scale) ** 3, self.rho * 1.0# g/m^3
        self.p_mass = self.p_vol * self.p_rho
        self.eps = 1e-5
        self.damping = 32.0
        self.x_0 = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=True)  # position
        self.v_0 = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=True)  # velocity

        # contact model parameters
        self.external_f = ti.Vector.field(3, dtype=float, shape=(self.sub_steps), needs_grad=True) # accumulated force from each particle
        self.cache = dict() # for grad backward

    def init(self, position, orientation, velocity):
        self.set_object_params(position, orientation, velocity)
        self.init_object()

    @ti.kernel
    def reset(self):
        self.external_f.fill(0.0)

    def set_object_params(self, position, orientation, velocity):
        self.init_pos[None] = position
        self.init_vel[None] = velocity

        rot = R.from_rotvec(np.deg2rad([orientation[0], orientation[1], orientation[2]]))
        self.init_rot = rot.as_matrix()
        self.rot_h[None] = self.init_rot
        self.t_h[None] = ti.Vector([position[0], position[1], position[2]])

    def set_object_pose(self, d_pos, d_ori):
        # add a **small** position/orientation correction
        rot = R.from_rotvec(np.deg2rad([d_ori[0], d_ori[1], d_ori[2]]))
        rot_mat = rot.as_matrix()

        self.drot_h[None] = np.matmul(rot_mat, self.init_rot)
        self.dt_h[None] = ti.Vector([self.init_pos[None][0] + d_pos[0], self.init_pos[None][1] + d_pos[1], self.init_pos[None][2] + d_pos[2]])

        self.set_particle_pose()

        self.rot_h[None] = self.drot_h[None]
        self.t_h[None] = self.dt_h[None]

    @ti.kernel
    def set_object_vel(self, dx:ti.f32, dy:ti.f32, dz:ti.f32, f:ti.i32):
        for i in range(self.n_particles):
            self.v_0[f, i] = ti.Vector([dx, dy, dz])

    @ti.kernel
    def set_particle_pose(self, f:ti.i32):
        for i in range(self.n_particles):
            init_t_pos = self.particles[i]

            before_t_pos = self.rot_h[None] @ init_t_pos + self.t_h[None]
            after_t_pos = self.drot_h[None] @ init_t_pos + self.dt_h[None]
            self.v_0[f, i] = (after_t_pos - before_t_pos)/(self.dt * self.sub_steps)

    @ti.kernel
    def init_object(self):
        for i in range(self.n_particles):
            before_t_pos = self.particles[i]
            after_t_pos = self.rot_h[None] @ before_t_pos + self.t_h[None]
            self.x_0[0, i] = after_t_pos
            self.v_0[0, i] = ti.Matrix([self.init_vel[None][0], self.init_vel[None][1], self.init_vel[None][2]])

    @ti.kernel
    def update(self, f:ti.i32):
        for p in range(self.n_particles):
            self.v_0[f+1, p] = self.v_0[f, p]
            self.x_0[f+1, p] = self.x_0[f, p] + self.dt * self.v_0[f, p]

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_particles):
            self.x_0[target, p] = self.x_0[source, p]
            self.v_0[target, p] = self.v_0[source, p]

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_particles):
            self.x_0.grad[target, p] = self.x_0.grad[source, p]
            self.v_0.grad[target, p] = self.v_0.grad[source, p]

    @ti.kernel
    def load_step_from_cache(self, f: ti.i32, cache_x_0: ti.types.ndarray(), cache_v_0: ti.types.ndarray()):
        for p in range(self.n_particles):
            for i in ti.static(range(self.dim)):
                self.x_0[f, p][i] = cache_x_0[p,i]
                self.v_0[f, p][i] = cache_v_0[p,i]

    @ti.kernel
    def add_step_to_cache(self, f: ti.i32, cache_x_0: ti.types.ndarray(), cache_v_0: ti.types.ndarray()):
        for p in range(self.n_particles):
            for i in ti.static(range(self.dim)):
                cache_x_0[p,i] = self.x_0[f, p][i]
                cache_v_0[p,i] = self.v_0[f, p][i]


    def memory_to_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.cache[cur_step_name] = dict()

        self.cache[cur_step_name]['x_0'] = torch.zeros((self.n_particles, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['v_0'] = torch.zeros((self.n_particles, self.dim), dtype=TC_TYPE, device=device)

        self.add_step_to_cache(0, self.cache[cur_step_name]['x_0'], self.cache[cur_step_name]['v_0'])
        self.copy_frame(self.sub_steps-1, 0)

    def memory_from_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.copy_frame(0, self.sub_steps-1)
        self.copy_grad(0, self.sub_steps-1)
        self.clear_step_grad(self.sub_steps-1)

        self.load_step_from_cache(0, self.cache[cur_step_name]['x_0'], self.cache[cur_step_name]['v_0'])

    @ti.kernel
    def clear_loss_grad(self):
        pass

    @ti.kernel
    def clear_step_grad(self, f:ti.i32):
        for p in range(self.n_particles):
            for t in range(f):
                self.x_0.grad[t,p].fill(0.0)
                self.v_0.grad[t,p].fill(0.0)
