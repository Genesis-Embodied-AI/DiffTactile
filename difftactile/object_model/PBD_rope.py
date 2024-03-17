"""
a class to describe soft cables with pbd
"""

import os
import taichi as ti
import torch
from math import pi
import numpy as np

TI_TYPE = ti.f32
TC_TYPE = torch.float32
NP_TYPE = np.float32

@ti.data_oriented
class PBDRope:
    def __init__(self, dt=5e-5, sub_steps=80, p_rad=0.05, p_rho=1.0, n_particles=10, table_height= 0.0, rest_length = None):
        self.sub_steps = sub_steps
        self.dt = dt
        ## the more ovelap the vertices have, the softer the cable is
        self.dim = 3
        self.p_rad = p_rad #
        self.p_rho = p_rho
        self.p_vol = ti.math.pi*4/3*(self.p_rad**3)
        self.p_mass = self.p_vol * self.p_rho
        self.invM = 1.0/self.p_mass
        self.gravity = ti.Vector([0.0, -9.8, 0.0])
        self.eps = 1e-10
        self.table_height = table_height

        self.n_particles = n_particles # number of particles
        self.n_segments = self.n_particles-1

        if rest_length is None:
            self.rest_length = 2 * self.p_rad # the rest length of each segment
            self.bending_length = 4 * self.p_rad # the distance between two sets of segments
            self.length = self.rest_length * self.n_particles # the length of the rope
        else:
            self.rest_length = rest_length # the rest length of each segment
            self.bending_length = 2 * rest_length # the distance between two sets of segments
            self.length = self.rest_length * (self.n_particles - 1) + self.p_rad*2 # the length of the rope

        self.stretch_compliance = ti.field(dtype=ti.f32, shape=())
        self.bending_compliance = ti.field(dtype=ti.f32, shape=())
        self.stretch_relaxation = ti.field(dtype=ti.f32, shape=())
        self.bending_relaxation = ti.field(dtype=ti.f32, shape=())

        self.stretch_compliance[None] = 1e-5 / self.p_rho
        self.bending_compliance[None] = 1e-5 / self.p_rho
        self.stretch_relaxation[None] = 0.3
        self.bending_relaxation[None] = 0.1

        self.init_ori = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.target_ori = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.init_pos = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_particles))  # init_position
        self.target_pos = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_particles))  # target_position
        self.tmp_pos = ti.Vector.field(3, dtype=ti.f32, shape=(self.sub_steps, self.n_particles), needs_grad=True)
        self.tmp_vel = ti.Vector.field(3, dtype=ti.f32, shape=(self.sub_steps, self.n_particles), needs_grad=True)
        self.dpos = ti.Vector.field(3, dtype=ti.f32, shape=(self.sub_steps, self.n_particles), needs_grad=True)

        self.pos = ti.Vector.field(3, dtype=ti.f32, shape=(self.sub_steps, self.n_particles), needs_grad=True)  # position
        self.vel = ti.Vector.field(3, dtype=ti.f32, shape=(self.sub_steps, self.n_particles), needs_grad=True)  # velocity
        self.ext_f = ti.Vector.field(3, dtype=ti.f32, shape=(self.sub_steps, self.n_particles), needs_grad=True) # external contact force

        self.floor_normal = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.floor_normal[None] = ti.Vector([0.0, 1.0, 0.0])
        self.floor_friction = 0.995
        self.cache = dict()

        #calculate travel distance
        self.travel_info = ti.Vector.field(1, dtype=ti.f32, shape=(self.sub_steps,self.n_particles), needs_grad=True)

    @ti.kernel
    def init(self, pos:ti.types.vector(3, ti.f32), ori:ti.types.vector(3, ti.f32)):

        # init pos
        self.init_pos[0] = pos # the fixed point
        self.init_ori[None] = ori

        for i in range(1, self.n_particles):
            self.init_pos[i] = self.init_pos[0] + self.init_ori[None] * self.rest_length * i

        for i in range(self.n_particles):
            self.pos[0, i] = self.init_pos[i]
            self.vel[0, i] = ti.Vector([0.0, 0.0, 0.0])
        self.travel_info.fill(0.0)

    @ti.kernel
    def set_target(self, pos:ti.types.vector(3, ti.f32)):

        self.target_pos[0] = pos #the fixed point
        whole_length = self.rest_length * (self.n_particles -1)
        height = pos[1]
        theta = ti.asin( height / whole_length)
        self.target_ori[None] = [0, -ti.sin(theta), ti.cos(theta)]
        for i in range(1, self.n_particles):
            self.target_pos[i] = self.target_pos[0] + self.target_ori[None] * self.rest_length * i


    @ti.kernel
    def reset(self):
        self.tmp_vel.fill(0.0)
        self.tmp_pos.fill(0.0)
        self.dpos.fill(0.0)
        self.ext_f.fill(0.0)

    @ti.func
    def update_contact_force(self, ext_v1, f, p):
        self.ext_f[f, p] += ext_v1 * self.dt

    @ti.kernel
    def update(self, f:ti.i32):
        ## update with external forces
        for i in range(self.n_particles):
            next_vel = self.vel[f, i]
            next_vel += self.ext_f[f, i] / self.p_mass
            next_vel += self.gravity * self.dt
            next_pos = self.pos[f, i] + next_vel * self.dt

            ## boundary condition
            if next_pos[1] < self.table_height and next_vel[1] < 0.0:

                #### remove the normal direction's velocity and add friction to the tangantial direction
                v_n = self.floor_normal[None].dot(next_vel) * self.floor_normal[None]
                v_t = next_vel - v_n
                next_vel = v_t * self.floor_friction

            self.tmp_vel[f, i] = next_vel
            self.tmp_pos[f, i] = self.pos[f, i] + self.tmp_vel[f, i] * self.dt

        ## solve stretch constrain
        for e in range(self.n_segments):
            x1 = self.tmp_pos[f, e]
            x2 = self.tmp_pos[f, e+1]

            w1 = self.invM
            w2 = self.invM

            n = x1 - x2
            n_norm = n.norm(self.eps)
            C = n_norm - self.rest_length
            alpha = self.stretch_compliance[None] / (self.dt**2)

            dp = -C / (w1 + w2 + alpha) * n / n_norm * self.stretch_relaxation[None]
            self.dpos[f, e] += dp * w1
            self.dpos[f, e+1] -= dp * w2

        ## update pos with the correction
        for i in range(self.n_particles):
            self.tmp_pos[f, i] += self.dpos[f, i]
            self.dpos[f, i] = [0.0, 0.0, 0.0]

        ## solve bending constrain
        for e in range(self.n_segments-1):
            x1 = self.tmp_pos[f, e]
            x2 = self.tmp_pos[f, e+2]

            w1 = self.invM
            w2 = self.invM

            n = x1 - x2
            n_norm = n.norm(self.eps)
            C = n_norm - self.bending_length
            alpha = self.bending_compliance[None] / (self.dt**2)

            dp = -C / (w1 + w2 + alpha) * n / n_norm * self.bending_relaxation[None]
            self.dpos[f, e] += dp * w1
            self.dpos[f, e+2] -= dp * w2

        ## update pos with the correction
        for i in range(self.n_particles):
            self.tmp_pos[f, i] += self.dpos[f, i]

        # advection
        for i in range(self.n_particles):
            if i == 0: # controlled point
                self.pos[f+1, i] = self.init_pos[i]
            else:
                self.pos[f+1, i] = self.tmp_pos[f, i]
            self.vel[f+1, i] = (self.pos[f+1, i] - self.pos[f, i]) / self.dt

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_particles):
            self.pos[target, p] = self.pos[source, p]
            self.vel[target, p] = self.vel[source, p]
            self.travel_info[target, p] = self.travel_info[source, p]

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_particles):
            self.pos.grad[target, p] = self.pos.grad[source, p]
            self.vel.grad[target, p] = self.vel.grad[source, p]
            self.travel_info.grad[target, p] = self.travel_info.grad[source, p]

    @ti.kernel
    def load_step_from_cache(self, f: ti.i32, cache_pos: ti.types.ndarray(), cache_vel: ti.types.ndarray(), cache_travel_info: ti.types.ndarray()):
        for p in range(self.n_particles):
            for i in ti.static(range(self.dim)):
                self.pos[f, p][i] = cache_pos[p,i]
                self.vel[f, p][i] = cache_vel[p,i]
        for p in range(self.n_particles):
            self.travel_info[f, p][0] = cache_travel_info[p, 0]

    @ti.kernel
    def add_step_to_cache(self, f: ti.i32, cache_pos: ti.types.ndarray(), cache_vel: ti.types.ndarray(), cache_travel_info: ti.types.ndarray()):
        for p in range(self.n_particles):
            for i in ti.static(range(self.dim)):
                cache_pos[p,i] = self.pos[f, p][i]
                cache_vel[p,i] = self.vel[f, p][i]

        for p in range(self.n_particles):
            cache_travel_info[p, 0] = self.travel_info[f, p][0]

    def memory_to_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.cache[cur_step_name] = dict()

        self.cache[cur_step_name]['pos'] = torch.zeros((self.n_particles, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['vel'] = torch.zeros((self.n_particles, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['travel_info'] = torch.zeros((self.n_particles, 1), dtype=TC_TYPE, device=device)
        self.add_step_to_cache(0, self.cache[cur_step_name]['pos'], self.cache[cur_step_name]['vel'], self.cache[cur_step_name]['travel_info'])
        self.copy_frame(self.sub_steps-1, 0)

    def memory_from_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.copy_frame(0, self.sub_steps-1)
        self.copy_grad(0, self.sub_steps-1)
        self.clear_step_grad(self.sub_steps-1)
        self.load_step_from_cache(0, self.cache[cur_step_name]['pos'], self.cache[cur_step_name]['vel'],  self.cache[cur_step_name]['travel_info'])

    @ti.kernel
    def clear_loss_grad(self):
        pass

    @ti.kernel
    def clear_step_grad(self, f:ti.i32):
        self.tmp_pos.grad.fill(0.0)
        self.tmp_vel.grad.fill(0.0)
        self.dpos.grad.fill(0.0)
        self.ext_f.grad.fill(0.0)
        for p in range(self.n_particles):
            for t in range(f):
                self.pos.grad[t, p].fill(0.0)
                self.vel.grad[t, p].fill(0.0)
                self.travel_info.grad[t, p].fill(0.0)
