import os
import pybullet
from pybullet_envs.robot_bases import XmlBasedRobot
from custom_envs import ROBOTS_DIR

class MJCFBasedRobot(XmlBasedRobot):
    """ 
        This class is a copy of the Pybullet implementation that changes the XML file directory
        """

    def __init__(self, model_xml, robot_name, action_dim, obs_dim, self_collision=True):
        XmlBasedRobot.__init__(self, robot_name, action_dim, obs_dim, self_collision)
        self.model_xml = model_xml
        self.doneLoading = 0

    # TODO figure out how relative file paths work so that I don't have to put the full path here like an idiot
    def reset(self, bullet_client):
        modelPath = os.path.join(ROBOTS_DIR, 'mjcf', self.model_xml)
        self._p = bullet_client
        if (self.doneLoading == 0):
            self.ordered_joints = []
            self.doneLoading = 1
            if self.self_collision:
                """
                self.objects = self._p.loadMJCF(modelFilePath,
                                                flags=pybullet.URDF_USE_SELF_COLLISION |
                                                pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
                                                pybullet.URDF_GOOGLEY_UNDEFINED_COLORS )
                """ 
                self.objects = self._p.loadMJCF(modelPath,
                                                flags=pybullet.URDF_USE_SELF_COLLISION |
                                                pybullet.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS |
                                                pybullet.MJCF_COLORS_FROM_FILE)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p, self.objects)
            else:
                self.objects = self._p.loadMJCF(modelPath, flags = pybullet.URDF_GOOGLEY_UNDEFINED_COLORS)
                self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
                    self._p, self.objects)
        
        self.robot_specific_reset(self._p)
        s = self.calc_state(
        )  # optimization: calc_state() can calculate something in self.* for calc_potential() to use

        return s

    def calc_potential(self):
        return 0
