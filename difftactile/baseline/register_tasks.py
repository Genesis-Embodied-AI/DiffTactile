""" 
    register_tasks.py
    Implemented by Elgce August, 2023
    register tasks with different parameters for use of ppo_train
    For use of stable-baselines3, both PPO & SAC
"""
from difftactile.tasks.box_open             import Contact as box_open_contact
from difftactile.tasks.object_repose        import Contact as repose_obj_contact
from difftactile.tasks.surface_follow       import Contact as surface_follow_contact
from difftactile.tasks.cable_straightening  import Contact as backward_pbd_contact

##################################################################
# A class to define different tasks
##################################################################
class ContactTask():
    def __init__(self, task_name, contact_model, n_actions, tactile_n_observations, state_n_observations, dt, total_steps, sub_steps, obj, x0=None):
        self.dt = dt
        self.x0 = x0
        self.obj = obj
        self.n_actions = n_actions
        self.sub_steps = sub_steps        
        self.task_name = task_name
        self.total_steps = total_steps
        self.contact_model = contact_model
        self.state_n_observations = state_n_observations
        self.tactile_n_observations = tactile_n_observations
##################################################################

##################################################################
# register all taichi tasks & get task info here
##################################################################
def register_tasks():
    return {
        "surface_follow": ContactTask(
            task_name = "surface_follow",
            contact_model = surface_follow_contact,
            n_actions = 6,
            tactile_n_observations = 6 + 3 + 2 * 136,
            state_n_observations = 6,
            dt = 5e-4,
            total_steps = 1000,
            sub_steps = 50,
            obj = "Random-surface.stl",
            x0 = [0.0, 0.075, 0.0, 0.0, 0.0, 0.0] * 15 + [-0.1, 0.0, 0.0, 0.0, 0.0, 0.0] * 985
        ),
        "repose_obj": ContactTask(
            task_name = "repose_obj",
            contact_model = repose_obj_contact,
            n_actions = 6,
            tactile_n_observations = 2 * 136 + 4 * 136 * 3 + 3 + 6,
            state_n_observations = 2 * 136 * 2 * 3 + 6,
            dt = 5e-5,
            total_steps = 600,
            sub_steps = 50,
            obj = "block-10.stl",
            x0 = [0.0, 1.5, 0.0, 0.0, 0.0, 0.0] * 300 + [1.0, 0.5, 0.0, 0.0, 0.0, 0.0] * 300
        ),
        "cable_manip": ContactTask(
            task_name = "cable_manip",
            contact_model = backward_pbd_contact,
            n_actions = 7,
            tactile_n_observations = 2 * 136 + 50 * 3 + 6 + 3,
            state_n_observations = 50 * 3 + 6,
            dt = 5e-4,
            total_steps = 800,
            sub_steps = 50,
            obj = None,
            x0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8] * 200 + [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0] * 600
        ),
        "box_open": ContactTask(
            task_name = "box_open",
            contact_model = box_open_contact,
            n_actions = 6,
            tactile_n_observations = 2*136 + 4*136* 3 + 3 + 6,
            state_n_observations = 2 * 136 * 2 * 3 + 6,
            dt = 5e-5,
            total_steps = 600,
            sub_steps = 50,
            obj = "earpod-case.stl",
            x0 = [0.0, 0.8, 0.0, 0.0, 0.0, 0.0] * 150 + [0.8, 0.0, 0.0, 0.0, 0.0, 0.0] * 450
        )
        # NOTE: add more tasks to train with ppo & SAC
    }

def get_tasks(task_type):
    registered_task = register_tasks()
    if task_type not in registered_task.keys():
        print("Error: task "+ task_type + "has not been registered!")
        print("You can choose from below registered tasks:")
        tasks = []
        for item in registered_task.keys():
            tasks.append(item)
        max_length = max(len(task) for task in tasks)
        print('-' * (max_length + 4))
        for task in tasks:
            print('|', task.ljust(max_length), '|')
        print('-' * (max_length + 4))
        return None
    else:
        return registered_task[task_type]
#################################################################