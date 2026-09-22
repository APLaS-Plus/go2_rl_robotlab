# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the functions that are specific for the locomotion environments."""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

# Re-export the Go2 task MDP helpers so that the Lite3 task can reuse the exact
# same reward/observation implementations without duplicating code.
from robot_lab.tasks.go2.mdp.commands import *  # noqa: F401, F403
from robot_lab.tasks.go2.mdp.curriculums import *  # noqa: F401, F403
from robot_lab.tasks.go2.mdp.events import *  # noqa: F401, F403
from robot_lab.tasks.go2.mdp.observations import *  # noqa: F401, F403
from robot_lab.tasks.go2.mdp.rewards import *  # noqa: F401, F403
from robot_lab.tasks.go2.mdp.utils import *  # noqa: F401, F403

from .terrains import *  # noqa: F401, F403
