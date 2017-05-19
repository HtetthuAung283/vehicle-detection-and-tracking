import numpy as np
import copy

# a class for tracking the detected objects
class Vehicle():
    def __init__(self, position):

        # search radius between anticipated center and detected position
        self.radius = 20
        
        # the current position of the vehicle
        self.position = position

        # the past positions of the vehicle
        self.history_positions = []
        
        # last confirmation failed
        self.confirmation = True

    def confirmVehicle(self, positions):
        '''
            confirm this vehicle by going through detected positions
            if a position is found in a certain radius around the anticipated position,
            then this position is the new position of this vehicle
        '''
        anticipatedPosition = self.anticipatePosition()
        
        newConfirmation = False
        
        distancePosition = {}
        
        for testPosition in positions:
            distancePosition[self.position.distance(testPosition)] = testPosition
        
        print('positions:', distancePosition)
        
        # if the nearest position is < than the search radius, then we have a match
        # this position is the new position of this vehicle
        distancesList = list(distancePosition.keys())
        distancesList.sort()
        if len(distancesList) > 0:
            print('sorted list of distances', distancesList)
            if distancesList[0] < self.radius:
                newConfirmation = True
                
                # put the old position into history
                self.history_positions.append(self.position)
                
                # renew position
                self.position = distancePosition[distancesList[0]]
                positions.remove(distancePosition[distancesList[0]])
        
        # return the position list
        if self.confirmation and not newConfirmation:
            self.confirmation = False
            return True, positions
        elif not self.confirmation and not newConfirmation:
            return False, positions
        elif not self.confirmation and newConfirmation:
            self.confirmation = True
            return True, positions
        
    def anticipateMovement(self):
        '''
            the vector that points from current position to the next likely position is calculated from
            the last 3 movementvectors with weighing the most recent vector more than the older ones:
            anticipated_Vector = (0.3 * movementvector[-3] + 0.5 * movementvector[-2] + 1 * movementvector[-1]) / 1.8 
        '''

        move_minus_3 = (0, 0)
        move_minus_2 = (0, 0)
        move_minus_1 = (0, 0)
        
        if len(self.history_positions) > 2:
            move_minus_3 = self.getMovementOfStep(-3)
        if len(self.history_positions) > 1:
            move_minus_2 = self.getMovementOfStep(-2)
        if len(self.history_positions) > 0:
            move_minus_1 = self.getMovementOfStep(-1)

        return (
                ( move_minus_3[0] * 0.3 + move_minus_2[0] * 0.5 + move_minus_1[0] ) / 1.8,
                ( move_minus_3[1] * 0.3 + move_minus_2[1] * 0.5 + move_minus_1[1] ) / 1.8
                )

    def anticipatePosition(self):
        movement = self.anticipateMovement()
        nextPosition = copy.deepcopy(self.position)
        nextPosition.x += movement[0]
        nextPosition.y += movement[1]
    
        return nextPosition

    def getBoundingBox(self):
        x1 = int(self.position.x - ((self.position.w + self.history_positions[-1].w + self.history_positions[-2].w) / 3) / 2)
        y1 = int(self.position.y - ((self.position.h + self.history_positions[-1].h + self.history_positions[-2].h) / 3) / 2)
        x2 = int(self.position.x + ((self.position.w + self.history_positions[-1].w + self.history_positions[-2].w) / 3) / 2)
        y2 = int(self.position.y + ((self.position.h + self.history_positions[-1].h + self.history_positions[-2].h) / 3) / 2)
    
        return ( (x1, y1) , (x2, y2) )
    
# GETTER
    def getMovementOfStep(self, pastTimeStep):
        
        if pastTimeStep == -1:
            return (self.position.x - self.history_positions[-1].x, self.position.y - self.history_positions[-1].y)
        elif pastTimeStep < -1:
            return (self.history_positions[pastTimeStep + 1].x - self.history_positions[pastTimeStep].x, self.history_positions[pastTimeStep + 1].y - self.history_positions[pastTimeStep].y)
        else:
            print('pastTimeStep has to be smaller than -1. you set it to', pastTimeStep)
            sys.exit()

    