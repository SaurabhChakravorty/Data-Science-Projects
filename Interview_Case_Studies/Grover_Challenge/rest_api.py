##########################################################################################
# We have used Flask API for writing test cases 
# The flask server will give the end point for REST API services
##########################################################################################


# import the libraries
import numpy as np
from flask import Flask, request
from flask_restful import Resource, Api
from main_code import simulation

# initialise the flask
app = Flask(__name__)
api = Api(app)

class test_cases(Resource):
    def __init__(self, sample_size = 50, robot_pos = (14, 2) , dino_pos = (15, 2) , direction = "top"):
        # make it class variables
        self.sample_size = sample_size
        self.robot_pos   = robot_pos
        self.dino_pos    = dino_pos
        self.direction   = direction
    
    def get(self):
        # empty simulation space
        s = simulation(sample_size=self.sample_size)

        # mark 'Robot' position
        s.agent_position(agent = "robot", position = self.robot_pos)

        # mark 'Dinosaur' position
        s.agent_position(agent = "dinosaur", position = self.dino_pos)
        
        # check if robot is moving
        s.move_robot(direction = self.direction)
        
        # kill the "dino"
        s.attack_dino()
        
        # display the current state
        s.__display_position__()
              
api.add_resource(test_cases, '/')

if __name__ == '__main__':
    app.run(debug=True)

