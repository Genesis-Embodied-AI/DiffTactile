import taichi as ti
import os
import numpy as np
from difftactile.object_model.mpm_plastic import MPMObj


@ti.data_oriented
class Deform_Loss:
    def __init__(self, sim:MPMObj):
        dtype = self.dtype = sim.dtype
        self.res = (sim.n_grid, sim.n_grid, sim.n_grid)
        self.n_grid = sim.n_grid
        self.dx_0 = sim.dx_0
        self.dim = sim.dim
        self.sub_steps = sim.sub_steps
        self.n_particles = sim.n_particles

        self.grid_mass = sim.grid_m_s
        self.particle_x = sim.x_0

        self.compute_grid_mass = sim.compute_grid_m_kernel
        self.target_density = ti.field(dtype=dtype, shape=self.res)
        self.target_sdf = ti.field(dtype=dtype, shape=self.res)
        self.nearest_point = ti.Vector.field(self.dim, dtype=dtype, shape=self.res)
        self.target_sdf_copy = ti.field(dtype=dtype, shape=self.res)
        self.nearest_point_copy = ti.Vector.field(self.dim, dtype=dtype, shape=self.res)
        self.inf = 1000
        self.sdf_loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.density_loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.loss = ti.field(dtype=dtype, shape=(), needs_grad=True)
        self.sdf_weight = ti.field(dtype=dtype, shape=())
        self.density_weight = ti.field(dtype=dtype, shape=())
        self.cur_iou = ti.field(dtype=dtype, shape=())
        self.target_iou = ti.field(dtype=dtype, shape=())



    def load_target_density(self, grids=None):
        grids = grids.to_numpy()
        self.target_density.from_numpy(grids)
        self.update_target()
        self.grid_mass.from_numpy(grids)
        self.iou()
        self.target_iou = self.cur_iou

    def initialize(self, grids=None):
        self.sdf_weight[None] = 1.0
        self.density_weight[None] = 1.0
        self.cur_iou.fill(0)
        self.load_target_density( grids)

    def set_weights(self, sdf, density):
        self.sdf_weight[None] = sdf
        self.density_weight[None] = density


    @ti.func
    def norm(self, x, eps=1e-8):
        return ti.sqrt(x.dot(x) + eps)

    @ti.kernel
    def update_target_sdf(self):
        for I in ti.grouped(self.target_sdf):
            self.target_sdf[I] = self.inf
            grid_pos = ti.cast(I * self.dx_0, self.dtype)
            if self.target_density[I] > 1e-4:
                self.target_sdf[I] = 0.
                self.nearest_point[I] = grid_pos
            else:
                for offset in ti.grouped(ti.ndrange(*(((-3, 3),)*self.dim))):
                    v = I + offset
                    if v.min() >= 0 and v.max() < self.n_grid and ti.abs(offset).sum() != 0:
                        if self.target_sdf_copy[v] < self.inf:
                            nearest_point = self.nearest_point_copy[v]
                            dist = self.norm(grid_pos - nearest_point)
                            if dist < self.target_sdf[I]:
                                self.nearest_point[I] = nearest_point
                                self.target_sdf[I] = dist
        for I in ti.grouped(self.target_sdf):
            self.target_sdf_copy[I] = self.target_sdf[I]
            self.nearest_point_copy[I] = self.nearest_point[I]

    def update_target(self):
        self.target_sdf_copy.fill(self.inf)
        for i in range(self.n_grid * 2):
            self.update_target_sdf()


    @ti.func
    def soft_weight(self, d):
        return 1/(1+d*d*10000)


    @ti.kernel
    def compute_density_loss_kernel(self):
        for I in ti.grouped(self.grid_mass):
            self.density_loss[None] += ti.abs(self.grid_mass[I] - self.target_density[I])

    @ti.kernel
    def compute_sdf_loss_kernel(self):
        for I in ti.grouped(self.grid_mass):
            self.sdf_loss[None] += self.target_sdf[I] * self.grid_mass[I]
    @ti.kernel
    def sum_up_loss_kernel(self):
        self.loss[None] += self.density_loss[None] * self.density_weight[None]
        self.loss[None] += self.sdf_loss[None] * self.sdf_weight[None]




    @ti.kernel
    def clear_loss_grad(self):
        self.loss[None] = 0
        self.density_loss[None] = 0
        self.sdf_loss[None] = 0

        self.density_loss.grad[None] = 0
        self.sdf_loss.grad[None] = 0

    def compute_loss_kernel(self, f:ti.i32):

        self.grid_mass.fill(0)
        self.compute_grid_mass(f)

        self.compute_density_loss_kernel()
        self.compute_sdf_loss_kernel()
        self.sum_up_loss_kernel()

    def compute_loss_kernel_grad(self, f):
        self.sum_up_loss_kernel.grad()
        self.grid_mass.fill(0.)
        self.grid_mass.grad.fill(0.)
        self.compute_grid_mass(f) # get the grid mass tensor...
        self.compute_sdf_loss_kernel.grad()
        self.compute_density_loss_kernel.grad()
        self.compute_grid_mass.grad(f) # back to the particles..

    @ti.kernel
    def iou(self):
        ma = ti.cast(0., self.dtype)
        mb = ti.cast(0., self.dtype)
        I = ti.cast(0., self.dtype)
        Ua = ti.cast(0., self.dtype)
        Ub = ti.cast(0., self.dtype)
        for i in ti.grouped(self.grid_mass):
            ti.atomic_max(ma, self.grid_mass[i])
            ti.atomic_max(mb, self.target_density[i])
            I += self.grid_mass[i]  * self.target_density[i]
            Ua += self.grid_mass[i]
            Ub += self.target_density[i]
        I = I/ma/mb
        U = Ua/ma + Ub/mb
        self.cur_iou[None] = I/(U- I)
     

    def iou2(self, a, b):
        I = np.sum(a * b)
        return I / (np.sum(a) + np.sum(b) - I)

   