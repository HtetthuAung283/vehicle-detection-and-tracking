import math

# a class for tracking the detected objects
class Position():
    def __init__(self, x_center, y_center, height, width):

        # x of center
        self.x = x_center
        
        # y of center
        self.y = y_center
        
        # height
        self.h = height
        
        # width
        self.w = width

    def distance(self, position):
        '''
            calculate the distance between this and the given position
        '''
        distance = math.sqrt((self.x - position.x) ** 2 + (self.y - position.y) ** 2)
        
        return distance
