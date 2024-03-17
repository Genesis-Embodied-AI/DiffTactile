"""
a class to describe soft elasto-plastic objects with mpm
"""
import os
import taichi as ti
import torch
from math import pi
import numpy as np
from scipy.spatial.transform import Rotation as R
from difftactile.object_model.obj_loader import ObjLoader

TI_TYPE = ti.f32
TC_TYPE = torch.float32
NP_TYPE = np.float32

@ti.data_oriented
class MPMObj:
    def __init__(self, dt=5e-5, sub_steps=80, obj_name=None, space_scale = 1.0, obj_scale = 1.0, density = 2, rho = 1):
        '''
        we assume the base space is 1 m x 1 m x 1 m
        space scale is to enlarge or shrink the manipulation space
        object scale is to adjust object's size, since we load all objects with scale 1
        '''
        self.sub_steps = sub_steps
        self.dt = dt
        self.ball_pos = ti.Vector.field(3,dtype=ti.f32, shape=())
        self.ball_ori = ti.Vector.field(3,dtype=ti.f32, shape=())
        self.ball_vel = ti.Vector.field(3,dtype=ti.f32, shape=())
        self.rot_h = ti.Matrix.field(3, 3, ti.f32, shape = ())
        self.trans_h = ti.Matrix.field(4, 4, ti.f32, shape = ())

        self.dim = 3
        self.bound = 3
        self.n_grid = 64
        self.space_scale = space_scale
        self.obj_scale = obj_scale
        self.rho = rho * self.obj_scale
        self.particle_density = self.n_grid * density * obj_scale / space_scale
        self.gravity = ti.Vector([0.0, -9.80, 0.0]) # m/s^2

        ## parameters for object
        self.obj_name = obj_name
        if self.obj_name is not None:
            data_path = os.path.join("..", "meshes", "objects", self.obj_name)
            obj_loader = ObjLoader(data_path, particle_density = int(self.particle_density))
            obj_loader.generate_particles()
            self.n_particles = len(obj_loader.particles)
            self.particles = ti.Vector.field(3, dtype=float, shape=self.n_particles)
            self.particles.from_numpy((obj_loader.particles * self.obj_scale).astype(np.float32))
            print("Object model is loaded!")
        else:
            print("ERR on loading object model!!")


        self.dx_0 = float(self.space_scale / self.n_grid)
        self.inv_dx_0 =  1 / self.dx_0
        self.p_vol, self.p_rho =  (self.dx_0 * self.obj_scale) ** 3, self.rho * 1.0# g/m^3
        self.p_mass = self.p_vol * self.p_rho
        self.eps = 1e-5
        self.dtype = ti.f32

        self.damping = 35.0

        self.E_0 = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.nu_0 = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.E_0[None], self.nu_0[None] = 4e3 * self.space_scale, 0.4  # Young's modulus and Poisson's ratio

        self.mu_0 = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.yield_stress = ti.field(dtype=ti.f32, shape=())
        self.lambda_0 = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.mu_0[None] = self.E_0[None] / 2 / (1 + self.nu_0[None])
        self.lambda_0[None] = self.E_0[None] * self.nu_0[None] / (1 + self.nu_0[None]) / (1 - 2 * self.nu_0[None])  # Lame parameters
        self.yield_stress[None] = 1000

        self.x_0 = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=True)  # position
        self.v_0 = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=True)  # velocity
        self.C_0 = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=True)  # affine velocity field
        self.F_new = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=True)
        self.F_0 = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=True)  # deformation gradient
        self.U_svd = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=True)
        self.V_svd = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=True)
        self.S_svd = ti.Matrix.field(3, 3, dtype=float, shape=(self.sub_steps, self.n_particles), needs_grad=True)

        self.grid_v_in = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid), needs_grad=True)  # grid node momentum/velocity
        self.grid_v_out = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid), needs_grad=True)  # grid node momentum/velocity
        self.grid_m = ti.field(dtype=float, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid), needs_grad=True)  # grid node mass
        self.grid_m_s = ti.field(dtype=float, shape=(self.n_grid, self.n_grid, self.n_grid), needs_grad=True)  # grid node mass
        self.grid_f = ti.Vector.field(3, dtype=float, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid), needs_grad=True)  # grid node external force
        self.grid_occupy = ti.field(dtype=int, shape=(self.sub_steps, self.n_grid, self.n_grid, self.n_grid))
        self.surf_f = ti.Vector.field(3, float, shape=(self.sub_steps), needs_grad = True) # surface aggreated 3-axis forces
        self.cache = dict() # for grad backward

    def init(self, position, orientation, velocity):
        self.set_object_params(position, orientation, velocity)
        self.init_object()

    def set_object_params(self, position, orientation, velocity):
        self.ball_pos[None] = position
        self.ball_ori[None] = orientation
        self.ball_vel[None] = velocity

        rot = R.from_rotvec(np.deg2rad([orientation[0], orientation[1], orientation[2]]))
        rot_mat = rot.as_matrix()
        trans_mat = np.eye(4)
        trans_mat[0:3,0:3] = rot_mat
        trans_mat[0,3] = position[0]; trans_mat[1,3] = position[1]; trans_mat[2,3] = position[2]
        self.rot_h[None] = rot_mat.tolist()
        self.trans_h[None] = trans_mat.tolist()

    @ti.kernel
    def init_object(self):
        for i in range(self.n_particles):
            before_t_pos = self.particles[i]
            after_t_pos = self.trans_h[None] @ ti.Vector([before_t_pos[0], before_t_pos[1], before_t_pos[2], 1.0]) # 4 x 1 homogeneous
            self.x_0[0,i] = ti.Vector([after_t_pos[0], after_t_pos[1], after_t_pos[2]])
            self.v_0[0,i] = ti.Matrix([self.ball_vel[None][0], self.ball_vel[None][1], self.ball_vel[None][2]])
            self.F_0[0,i] = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    @ti.kernel
    def reset(self):
        self.grid_v_in.fill(0.0)
        self.grid_v_out.fill(0.0)
        self.grid_m.fill(0.0)
        self.grid_m_s.fill(0.0)
        self.grid_f.fill(0.0)
        self.grid_occupy.fill(0)
        self.surf_f.fill(0.0)


    @ti.kernel
    def get_external_force(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.n_grid, self.n_grid, self.n_grid):
            self.surf_f[f] += self.grid_f[f, i, j, k] * self.dx_0

    @ti.func
    def update_contact_force(self, ext_v1, f, i, j, k):
        self.grid_f[f, i, j, k] += ext_v1 * self.dt

    @ti.func
    def norm(self, x , eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)

    @ti.func
    def make_matrix_from_diag(self, d):
        if ti.static(self.dim==2):
            return ti.Matrix([[d[0], 0.0], [0.0, d[1]]], dt=self.dtype)
        else:
            return ti.Matrix([[d[0], 0.0, 0.0], [0.0, d[1], 0.0], [0.0, 0.0, d[2]]], dt=self.dtype)

    @ti.func
    def compute_von_mises(self, F, U, sig, V, mu):
        epsilon = ti.Vector.zero(self.dtype, self.dim)
        sig = ti.max(sig, 0.05) # add this to prevent NaN in extrem cases
        if ti.static(self.dim == 2):
            epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1])])
        else:
            epsilon = ti.Vector([ti.log(sig[0, 0]), ti.log(sig[1, 1]), ti.log(sig[2, 2])])
        epsilon_hat = epsilon - (epsilon.sum() / self.dim)
        epsilon_hat_norm = self.norm(epsilon_hat)
        delta_gamma = epsilon_hat_norm - self.yield_stress[None] / (2 * mu)

        if delta_gamma > 0 :  # Yields
            epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
            sig = self.make_matrix_from_diag(ti.exp(epsilon))
            F = U @ sig @ V.transpose()
        return F

    @ti.kernel
    def compute_grid_m_kernel(self, f:ti.i32):
        for p in range(0, self.n_particles):
            base = (self.x_0[f, p] * self.inv_dx_0 - 0.5).cast(int)
            fx = self.x_0[f, p] * self.inv_dx_0 - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                weight = w[i][0] * w[j][1] * w[k][2]
                self.grid_m_s[ base + offset] += weight * self.p_mass

    @ti.kernel
    def compute_new_F(self, f: ti.i32):
        for p in range(self.n_particles):
            self.F_new[f, p] = (ti.Matrix.diag(dim=3, val=1) + self.dt * self.C_0[f, p]) @ self.F_0[f, p]

    @ti.kernel
    def svd(self, f: ti.i32):
        for p in range(self.n_particles):
            self.U_svd[f, p], self.S_svd[f, p], self.V_svd[f, p] = ti.svd(self.F_new[f, p])

    @ti.kernel
    def svd_grad(self, f: ti.i32):
        for p in range(self.n_particles):
            self.F_new.grad[f, p] += self.single_svd_grad(f, p)

    @ti.func
    def clamp(self, a: ti.f32):
        if a>=0:
            a = ti.max(a, 1e-8)
        else:
            a = ti.min(a, -1e-8)
        return a

    @ti.func
    def single_svd_grad(self, f: ti.i32, p: ti.i32):
        vt = self.V_svd[f, p].transpose()
        ut = self.U_svd[f, p].transpose()
        s_term = self.U_svd[f, p] @ self.S_svd.grad[f, p] @ vt

        s = ti.Vector.zero(ti.f32, 3)
        s = ti.Vector([self.S_svd[f, p][0, 0], self.S_svd[f, p][1, 1], self.S_svd[f, p][2, 2]]) ** 2
        ff = ti.Matrix.zero(ti.f32, 3, 3)
        for i, j in ti.static(ti.ndrange(3, 3)):
            if i == j:
                ff[i, j] = 0
            else:
                ff[i, j] = 1.0 / self.clamp(s[j] - s[i])
        u_term = self.U_svd[f, p] @ ((ff * (ut @ self.U_svd.grad[f, p] - self.U_svd.grad[f, p].transpose() @ self.U_svd[f, p])) @ self.S_svd[f, p]) @ vt
        v_term = self.U_svd[f, p] @ (self.S_svd[f, p] @ ((ff * (vt @ self.V_svd.grad[f, p] - self.V_svd.grad[f, p].transpose() @ self.V_svd[f, p])) @ vt))
        return u_term + v_term + s_term

    @ti.kernel
    def p2g(self, f:ti.i32):
        for p in range(self.n_particles): # Particle state update and scatter to grid (P2G)
            base = (self.x_0[f, p] * self.inv_dx_0 - 0.5).cast(int)
            fx = self.x_0[f, p] * self.inv_dx_0 - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_F = self.compute_von_mises(self.F_new[f,p], self.U_svd[f, p], self.S_svd[f, p], self.V_svd[f, p],  self.mu_0[None])

            J = new_F.determinant()

            r = self.U_svd[f, p] @ self.V_svd[f, p].transpose()
            cauchy = 2 * self.mu_0[None] * (new_F - r) @ new_F.transpose() + ti.Matrix.identity(float, 3) * self.lambda_0[None] * J * (J - 1)

            stress = (-self.dt * self.p_vol * 4 * self.inv_dx_0 * self.inv_dx_0) * cauchy

            # # FEM
            # F_i = self.F_new[f, p]
            # F_T = F_i.inverse().transpose()
            # J = F_i.determinant()
            # # J = ti.max(0.2, F_i.determinant())
            # log_J_i = ti.log(J)
            # cauchy = self.mu_0[None] * (F_i -  F_T) + self.lambda_0[None] * log_J_i * F_T
            # stress = (-self.dt * self.p_vol * 4 * self.inv_dx_0 * self.inv_dx_0) * cauchy

            affine = stress + self.p_mass * self.C_0[f, p]

            # Loop over 3x3 grid node neighborhood
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * self.dx_0
                weight = w[i][0] * w[j][1] * w[k][2]
                self.grid_v_in[f, base + offset] += weight * (self.p_mass * self.v_0[f, p] + affine @ dpos)
                self.grid_m[f, base + offset] += weight * self.p_mass

            self.F_0[f+1, p] = new_F

    @ti.kernel
    def check_grid_occupy(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.n_grid, self.n_grid, self.n_grid):
            if self.grid_m[f, i, j, k] > self.eps:
                self.grid_occupy[f, i, j, k] = 1

    @ti.kernel
    def grid_op(self, f:ti.i32):
        for i, j, k in ti.ndrange(self.n_grid, self.n_grid, self.n_grid):
            if self.grid_occupy[f, i, j, k] == 1:
                inv_m = 1 / (self.grid_m[f, i, j, k]+self.eps)

                v_out = ti.Vector([0.0, 0.0, 0.0])
                v_out += inv_m * self.grid_v_in[f, i, j, k] # Momentum to velocity
                v_out += inv_m * self.grid_f[f, i, j, k]
                v_out += self.dt * self.gravity  # gravity

                if i < self.bound and v_out[0] < 0:
                    v_out[0] = 0  # Boundary conditions
                if i > self.n_grid - self.bound and v_out[0] > 0:
                    v_out[0] = 0
                if j < self.bound and v_out[1] < 0: # < 3
                    v_out[1] = 0
                if j > self.n_grid - self.bound and v_out[1] > 0:
                    v_out[1] = 0
                if k < self.bound and v_out[2] < 0: # < 3
                    v_out[2] = 0
                if k > self.n_grid - self.bound and v_out[2] > 0:
                    v_out[2] = 0

                self.grid_v_out[f, i, j, k] = v_out


    @ti.kernel
    def g2p(self, f:ti.i32):
        for p in range(self.n_particles): # grid to particle (G2P)
            base = (self.x_0[f, p] * self.inv_dx_0 - 0.5).cast(int)
            fx = self.x_0[f, p] * self.inv_dx_0 - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, 3)
            new_C = ti.Matrix.zero(float, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                # loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j, k]).cast(float) - fx
                g_v = self.grid_v_out[f, base + ti.Vector([i, j, k])]
                weight = w[i][0] * w[j][1] * w[k][2]
                new_v += weight * g_v
                new_C += 4 * self.inv_dx_0 * weight * g_v.outer_product(dpos)

            self.v_0[f+1, p], self.C_0[f+1, p] = new_v, new_C
            self.x_0[f+1, p] = self.x_0[f, p] + self.dt * new_v  # advection

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_particles):
            self.x_0[target, p] = self.x_0[source, p]
            self.v_0[target, p] = self.v_0[source, p]
            self.C_0[target, p] = self.C_0[source, p]
            self.F_0[target, p] = self.F_0[source, p]

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_particles):
            self.x_0.grad[target, p] = self.x_0.grad[source, p]
            self.v_0.grad[target, p] = self.v_0.grad[source, p]
            self.C_0.grad[target, p] = self.C_0.grad[source, p]
            self.F_0.grad[target, p] = self.F_0.grad[source, p]

    @ti.kernel
    def load_step_from_cache(self, f: ti.i32, cache_x_0: ti.types.ndarray(), cache_v_0: ti.types.ndarray(), cache_C_0: ti.types.ndarray(), cache_F_0: ti.types.ndarray()):
        for p in range(self.n_particles):
            for i in ti.static(range(self.dim)):
                self.x_0[f, p][i] = cache_x_0[p,i]
                self.v_0[f, p][i] = cache_v_0[p,i]

            for i, j in ti.ndrange(self.dim, self.dim):
                self.C_0[f, p][i, j] = cache_C_0[p, i, j]
                self.F_0[f, p][i, j] = cache_F_0[p, i, j]

    @ti.kernel
    def add_step_to_cache(self, f: ti.i32, cache_x_0: ti.types.ndarray(), cache_v_0: ti.types.ndarray(), cache_C_0: ti.types.ndarray(), cache_F_0: ti.types.ndarray()):
        for p in range(self.n_particles):
            for i in ti.static(range(self.dim)):
                cache_x_0[p,i] = self.x_0[f, p][i]
                cache_v_0[p,i] = self.v_0[f, p][i]

            for i, j in ti.ndrange(self.dim, self.dim):
                cache_C_0[p, i, j] = self.C_0[f, p][i, j]
                cache_F_0[p, i, j] = self.F_0[f, p][i, j]

    def memory_to_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.cache[cur_step_name] = dict()

        self.cache[cur_step_name]['x_0'] = torch.zeros((self.n_particles, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['v_0'] = torch.zeros((self.n_particles, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['C_0'] = torch.zeros((self.n_particles, self.dim, self.dim), dtype=TC_TYPE, device=device)
        self.cache[cur_step_name]['F_0'] = torch.zeros((self.n_particles, self.dim, self.dim), dtype=TC_TYPE, device=device)

        self.add_step_to_cache(0, self.cache[cur_step_name]['x_0'], self.cache[cur_step_name]['v_0'], self.cache[cur_step_name]['C_0'], self.cache[cur_step_name]['F_0'])
        self.copy_frame(self.sub_steps-1, 0)

    def memory_from_cache(self, t):
        cur_step_name = f'{t:06d}'
        device = 'cpu'
        self.copy_frame(0, self.sub_steps-1)
        self.copy_grad(0, self.sub_steps-1)
        self.clear_step_grad(self.sub_steps-1)

        self.load_step_from_cache(0, self.cache[cur_step_name]['x_0'], self.cache[cur_step_name]['v_0'], self.cache[cur_step_name]['C_0'], self.cache[cur_step_name]['F_0'])

    @ti.kernel
    def clear_loss_grad(self):
        self.E_0.grad[None] = 0.0
        self.nu_0.grad[None] = 0.0
        self.mu_0.grad[None] = 0.0
        self.lambda_0.grad[None] = 0.0


    @ti.kernel
    def clear_step_grad(self, f:ti.i32):
        self.grid_v_in.grad.fill(0.0)
        self.grid_v_out.grad.fill(0.0)
        self.grid_m.grad.fill(0.0)
        self.grid_m_s.grad.fill(0.0)
        self.grid_f.grad.fill(0.0)
        self.F_new.grad.fill(0.0)
        self.U_svd.grad.fill(0.0)
        self.V_svd.grad.fill(0.0)
        self.S_svd.grad.fill(0.0)
        for p in range(self.n_particles):
            for t in range(f):
                self.x_0.grad[t,p].fill(0.0)
                self.v_0.grad[t,p].fill(0.0)
                self.C_0.grad[t,p].fill(0.0)
                self.F_0.grad[t,p].fill(0.0)
