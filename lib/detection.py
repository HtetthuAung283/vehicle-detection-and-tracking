import numpy as np
from vehicle import Vehicle

# a class for tracking the detected objects
class Detection():
    def __init__(self):
        
        # a vehicle should be plausible in n consecutive frames, to be accepted as a vehicle
        self.n = 3
        
        # all the positions of last frame
        self.positions = []
        
        # every item of this list contains the detected points of a single frame
        # the detected points are a list itself
        self.history_positions = []

        # vehicles, that are detected and traced over n frames with no fail
        self.vehicles = []


    def confirmExistentVehicles(self):
        
        confirmedVehicles = []
        
        for vehicle in self.vehicles:
            vehicleConfirmed, self.positions = vehicle.confirmVehicle(self.positions)
            
            # if existent vehicle has been confirmed
            if vehicleConfirmed:
                confirmedVehicles.append(vehicle)

        self.vehicles = confirmedVehicles


    def detectNewVehicles(self):
        '''
            use the history of uinassigned positions to detect a new vehicle
        '''
        
        print('detect new vehicles. history position length:', len(self.history_positions))
        
        if len(self.history_positions) < 3:
            return
        
        maybe_history_positions_minus_3 = []
        
        # start with all non-assigned positions 2 frames ago
        for position_minus_3 in self.history_positions[-3]:
            maybeVehicle = Vehicle(position_minus_3)
            
            maybeVehicleConfirmed_minus_3 = False
            
            # try to confirm this vehicle in 2 frames ago
            maybeVehicleConfirmed_minus_2, maybe_history_positions_minus_2 = maybeVehicle.confirmVehicle(self.history_positions[-2])
            
            # if vehicle has been confirmed
            if maybeVehicleConfirmed_minus_2:
                # try to confirm this vehicle in 1 frame ago
                maybeVehicleConfirmed_minus_1, maybe_history_positions_minus_1 = maybeVehicle.confirmVehicle(self.history_positions[-1])
        
                # if vehicle has been confirmed again, then this vehicle will be added to the detected vehicle list
                # and the old unassigned positions get updated
                if maybeVehicleConfirmed_minus_1:
                    # try to confirm this vehicle in current frame
                    maybeVehicleConfirmed, maybe_positions = maybeVehicle.confirmVehicle(self.positions)

                    self.vehicles.append(maybeVehicle)
                    self.history_positions[-2] = maybe_history_positions_minus_2
                    self.history_positions[-1] = maybe_history_positions_minus_1
                    self.positions = maybe_positions

            
            # if vehicle has not been confirmed through these 3 frames, then the old positions remain unchanged
            if not maybeVehicleConfirmed_minus_3:
                maybe_history_positions_minus_3.append(positions_minus_3)
        
        self.history_positions[-3] = maybe_history_positions_minus_3

    def detect(self):
        self.confirmExistentVehicles()
        self.detectNewVehicles()
        self.history_positions.append(self.positions)
        self.positions = []

# SETTER
    def addPosition(self, x_center, y_center, height, width):
        pos = {'x': x_center, 'y': y_center, 'h': height, 'w': width}
        self.positions.append(position)

    def addPositions(self, positions):
        self.positions.extend(positions)

# GETTER
    def getVehicleBoundingBoxes(self):
        bboxes = []
        
        for vehicle in self.vehicles:
            bboxes.append(vehicle.getBoundingBox())
     
        return bboxes
