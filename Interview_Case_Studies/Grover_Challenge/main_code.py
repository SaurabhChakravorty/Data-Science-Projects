##########################################################################################
# class function for processing data

# The documentation is given in the form of .ipynb file 

# It contains the class file which contains all modules for the simulation to take place
##########################################################################################


import numpy as np

class simulation:
    '''
    This class contains all the modules or methods for the simulation process
    '''
    
    def __init__(self, sample_size = 50):
        # create an numpy array of fixed size
        self.env = np.zeros((sample_size, sample_size))
        
        # intitialising the position of robot and dino
        self.robot = (0,0)
        self.env[self.robot] = 1.0 
        
        self.dino  = (1,1)
        self.env[self.dino] = 2.0 
        
        # for storing the location of "kill" of robot
        self.kill = []
        
    def agent_position(self, agent = "robot", position = (0,0)):     
        # changing the state based on condition
        if agent == "robot":
            env = self.robot
            mark = 1.0         
        else:
            env = self.dino
            mark = 2.0
            
        # assigning the position to agent 
        if self.__check_position__(position) and self.env[position] == 0.0 :
            self.env[env] = 0.0                      # setting the initial pos to 0.0
            if agent == "robot":
                self.robot = position
            else:
                self.dino = position
            self.env[position] = mark                # assigning Robot pos as 1.0 if condition is met          
        else:
            print("The position of {} cannot be updated in feature space".format(agent))    
            
    def move_robot(self, direction = "left"):
         # for moving the 'Robot' agent in simulation space
         if direction == 'left':
            self.robot = (self.robot[0] - 1, self.robot[1])
            
         if direction == 'right':
            self.robot = (self.robot[0] + 1, self.robot[1])
            
         if direction == 'top':
            self.robot = (self.robot[0], self.robot[1] + 1)
                          
         if direction == 'bottom':
            self.robot = (self.robot[0], self.robot[1] - 1)
       
         # check position and store the space
         self.agent_position(agent = "robot", position = self.robot)
         self.__display_position__()
            
    def attack_dino(self): 
        # attack in vicinity
        if np.abs(self.robot[0] - self.dino[0]) == 1 or np.abs(self.robot[1] - self.dino[1]) == 1 :
            self.robot = self.dino          # update the position
            self.env[self.robot] = 1.0     

            self.kill.append(self.robot)    # position where the kill happened
            self.dino = (0,0)               # initial pos again
        
            print("The dinosaur is killed at position {} and position of dinosaur is set to (0,0)".format(self.robot))
        else:
            print("We can't kill the dinosaur as it is far...")
                    
                    
    def __display_position__(self):
        # for displaying the positions of Dino and Robot
        print("The position of Robot is {} and position of Dinosaur is {}".format(self.robot, self.dino))
    
    
    def __check_position__(self, pos):
        # for checking the position in feature space
        if (0 <= pos[0] < self.env.shape[0]) and (0 <= pos[1] < self.env.shape[1]):
            return True
        else:
            return False
