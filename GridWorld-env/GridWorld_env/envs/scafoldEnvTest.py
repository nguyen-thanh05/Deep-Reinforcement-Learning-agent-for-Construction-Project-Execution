import unittest
import numpy as np
from grid_world import GridWorldEnv



class Action:
    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    UP = 4
    DOWN = 5
    PLACE_SCAFOLD = 6
    REMOVE_SCAFOLD = 7
    PLACE_FORWARD = 8
    PLACE_BACKWARD = 9
    PLACE_LEFT = 10
    PLACE_RIGHT = 11

action_enum = Action

class TestScaffoldEnv(unittest.TestCase):
    def setUp(self):
        self.env = GridWorldEnv(4, 1, debug=True)
        self.startPos = self.env.AgentsPos[0].copy()

    def test_invalidPlace(self):
        #place a block forward
        self.env.step((action_enum.PLACE_FORWARD, 0))

        # place scafold
        self.env.step((action_enum.PLACE_SCAFOLD, 0))

        # move up
        self.env.step((action_enum.UP, 0))

        # place a block forward
        self.env.step((action_enum.PLACE_FORWARD, 0))
        agentPos = self.env.AgentsPos[0]
        placed = self.env.building_zone[agentPos[0] + 1, agentPos[1], agentPos[2]] == -1

        np.testing.assert_array_equal(placed, 0)  # made sure agent did not place

    def test_invalidPlace2(self):
        agent_id = 0 
        pos = self.env.AgentsPos[agent_id].copy()
        # place a block front of the agent
        self.env.step((action_enum.PLACE_FORWARD, agent_id))

        # place a scaffold 
        self.env.step((action_enum.PLACE_SCAFOLD, agent_id))

        # walk left
        self.env.step((action_enum.LEFT, agent_id))

        # walk forward
        self.env.step((action_enum.FORWARD, agent_id))

        # place a scaffold
        self.env.step((action_enum.PLACE_SCAFOLD, agent_id))

        # walk forward
        self.env.step((action_enum.FORWARD, agent_id))

        # walk right
        self.env.step((action_enum.RIGHT, agent_id))

        # place a scafold
        self.env.step((action_enum.PLACE_SCAFOLD, agent_id))

        # walk right 
        self.env.step((action_enum.RIGHT, agent_id))

        # walk backward
        self.env.step((action_enum.BACKWARD, agent_id))

        # place a scafold
        self.env.step((action_enum.PLACE_SCAFOLD, agent_id))

        # climb up
        self.env.step((action_enum.UP, agent_id))

        # place a block to the left
        self.env.step((action_enum.PLACE_LEFT, agent_id))

        placed = self.env.building_zone[pos[0] + 1, pos[1], pos[2] + 1] == -1
        
        np.testing.assert_array_equal(placed, 1)  # made sure agent did placed

    def test_climbDown(self):
        # place a block forward
        self.env.step((action_enum.PLACE_FORWARD, 0))

        # place scafold
        self.env.step((action_enum.PLACE_SCAFOLD, 0))

        # move up
        self.env.step((action_enum.UP, 0))


        # place a scafold   
        self.env.step((action_enum.PLACE_SCAFOLD, 0))

        # move up
        self.env.step((action_enum.UP, 0))

        pos1 = self.env.AgentsPos[0].copy()

        # move forward
        self.env.step((action_enum.FORWARD, 0))

        pos2 = self.env.AgentsPos[0].copy()  # position of the agent after moving forward

        expectedPos = pos1.copy()
        expectedPos[0] += 1
        expectedPos[2] -= 1

        np.testing.assert_array_equal(expectedPos, pos2)

    def test_bug(self):
        # set agent position to [2, 2,0]
        self.env.AgentsPos[0] = np.array([2, 2, 0])
        # place right
        self.env.step((action_enum.PLACE_RIGHT, 0))

        self.assertTrue(1)

if __name__ == '__main__':
    unittest.main()

        
