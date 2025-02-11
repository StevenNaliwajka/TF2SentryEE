# NOT COMPLETE. In PROGRESS.

### To the ATF, It is structurally impossible for this design to be used with a real weapon and is not our goal. I am a law-abiding citizen.
### To Lockheed Martin, call me...
## Documentation is NOT complete. In progress
# TF2Sentry
This is the final piece of a project with [Maiikiru](https://github.com/Maiikiru) and Bradon
to create a fully functioning level one sentry from TF2 using an AIRSOFT gun.

Inbetween classes and work each of us have put countless hours into creating a cool project.

The work can be broken up into 3 repos.
- [Maiikiru's CV Repo](https://github.com/Maiikiru/TF2SentrySource) for the brain and the logic (PRIVATE for now)
- [Bradon's CAD Repo](https://github.com/ConvexBurrito5/TF2SentryCAD) for the physical CAD/STL files (PRIVATE for now)
- [Current Repo](https://github.com/ConvexBurrito5/TF2SentryEE) for circuitry and build instructions.

This repo is my work (Steven Naliwajka/ConvexBurrito5), [GitHub](https://github.com/ConvexBurrito5), [LinkedIn](https://www.linkedin.com/in/steven-naliwajka-69564929a/).

### Two options for targeting: 
1) Maiikiru has gone through and created a Computer vision (CV) repo that automatically detects and targets people: [Maiikiru's CV Repo](https://github.com/Maiikiru/TF2SentrySource)
2) I have also gone and created a remote control, [a slightly modified wrangler](https://www.youtube.com/watch?v=LYPzGNSfVRk)


To jump straight into building, see the Documentation folder.

![sentry](https://wiki.teamfortress.com/w/images/thumb/3/3a/TF2LVL1SG.png/163px-TF2LVL1SG.png)
![wrangler](https://wiki.teamfortress.com/w/images/thumb/2/27/BLU_Wrangler.png/192px-BLU_Wrangler.png)

## Review of my learned Skills
- At a high level, what we are trying to accomplish is simple however, as always, implementation is not as easy. I have learned lots:
  - Communication!
    - I2C Communication, learned how to implement and write code to utilize it in python.
    - Radio, utilizing 2.4Ghz wireless transceiver modules to talk from the turret to the wrangler
  - Power!
    - The turret and wrangler run off batteries.
    - 7.6v, 12v, 9.2v was all required. Got to use Buck DC voltage regulators.
    - Load calculation to make sure at least 5 hours of battery life.
  - Circuitry!
    - Circuitry to be able to plug in the turret without removing the battery. Same for Wrangler.
    - Performed proper wire size calculations; by working out max amps through each.
  - Coding!
    - Python - My first real project in python. I'm typically a java slob, so it was a little bit of a culture shock. But I'm glad I got to learn.
    - Arduino(C++) - Got to continue to hone my skills.
    - Git - I was dragged into the 21st century by Maiikiru. I'm proficient at Git now.
  - 3D Design!
    - Solidworks - I am competent at solidworks now.
  - Improved my 3D printing troubleshooting skills... and gotten a sizeable print farm out of it (2 decked out Ender 3s)
  - Working with Motors/Servos with control from both I2C and PWM.



With that out of the way, lets look into what we have completed.
## Features of the project
- Detection and tracking of a person, See [Maiikiru's CV Repo](https://github.com/Maiikiru/TF2SentrySource)
- We have provided [Maiikiru's CV Repo](https://github.com/Maiikiru/TF2SentrySource) with standard API interfaces so it easy to implement multiple hardware solutions. So far:
  - Raspberry Pi 5 (Please do not use a pi4, we tried to make it work, the performance is not good) , See 'instructionFT232H.md'
  - Standard laptop using [Adafruit's FT232H Breakout](https://www.adafruit.com/product/2264), See 'instructionRaspberryPi4.md'
- There are 4 interfaces implemented in each solution
  - FiringController
    - Handles the firing by pulling logic high on a switch to bridge the motor on the airsoft gun.
  - MotorController
    - Handles all motion. Basic set of methods to move your servos.
    - Idle motion, pans slowly looking for people.
  - PeripheralController
    - Handles the leds
  - SoundController
    - Handles all the sound, runs off of the devices default speaker.
- If a switch on the wrangler is turned on, automatic movement from [Maiikiru's CV Repo](https://github.com/Maiikiru/TF2SentrySource) will be disabled and overruled.
  - Input will be solely from the joystick and the firing button on the Wrangler.

## How can you build one?
Good news! Through many sleepless nights of documentation I have
compiled instructions to build your own TF2 Turret. You can begin by 
heading to the Documentation folder
