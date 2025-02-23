# #########
# # firstTry.py
# # This program is part of the online PS-Drone-API-tutorial on www.playsheep.de/drone.
# # It shows how to do basic movements with a Parrot AR.Drone 2.0 using the PS-Drone-API.
# # Dependencies: a POSIX OS, PS-Drone-API 2.0 beta or higher.
# # (w) J. Philipp de Graaff, www.playsheep.de, 2014
# ##########
# # LICENCE:
# #   Artistic License 2.0 as seen on http://opensource.org/licenses/artistic-license-2.0 (retrieved December 2014)
# #   Visit www.playsheep.de/drone or see the PS-Drone-API-documentation for an abstract from the Artistic License 2.0.
# ###########

# import time
# import ps_drone                # Imports the PS-Drone-API

# drone = ps_drone.Drone()       # Initializes the PS-Drone-API
# drone.startup()                # Connects to the drone and starts subprocesses

# drone.takeoff()                # Drone starts
# time.sleep(7.5)                # Gives the drone time to start

# drone.moveForward()            # Drone flies forward...
# time.sleep(2)                  # ... for two seconds
# drone.stop()                   # Drone stops...
# time.sleep(2)                  # ... needs, like a car, time to stop

# drone.moveBackward(0.25)       # Drone flies backward with a quarter speed...
# time.sleep(1.5)                # ... for one and a half seconds
# drone.stop()                   # Drone stops
# time.sleep(2)	

# drone.setSpeed(1.0)            # Sets default moving speed to 1.0 (=100%)
# print (drone.setSpeed())         # Shows the default moving speed

# drone.turnLeft()               # Drone moves full speed to the left...
# time.sleep(2)                  # ... for two seconds
# drone.stop()                   # Drone stops
# time.sleep(2)

# drone.land()                   # Drone lands

# ************************************************************* #
                        # Drone Take-off:

import time
import ps_drone                # Imports the PS-Drone-API

drone = ps_drone.Drone()       # Initializes the PS-Drone-API
drone.startup()                # Connects to the drone and starts subprocesses

drone.setSpeed(0.2)            # Sets default moving speed to 1.0 (=100%)
drone.takeoff()                # Drone starts
time.sleep(2.5)                # Gives the drone time to start

print("Drone taking off with a speed of: ", drone.setSpeed())

drone.stop()                     # Drone stops
time.sleep(2)

drone.land()                   # Drone lands