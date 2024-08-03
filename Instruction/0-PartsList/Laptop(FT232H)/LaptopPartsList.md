## List of all the parts needed:

1) Laptop/PC (1)- I ended up using an old lenovo Thinkpad that was tossed from work. Any Linux Based PC.
    - Brain of the Operation
    - $0 - To skew the final number im going to cheat and assume you are reading this on a PC. If not L33T and on linux already (Like me), look into dual booting.
2) FT232H Adafruit chip (1)
    - Used to convert USB -> GPIO/I2C
    - While I normally support buying the great value version.. Purchase the offical Adafruit FT232H. I could not get the off brand version to work.
    - $20: Amazon (1)
3) Arduino Uno (2)
    - 1 Handles I2C communication, 1 controls Wrangler
    - $14: Amazon (1)
    - $0.80 (1) + $3: AliExpress.
4) Relay Module (12V) (1)
    - Used for Control of Airsoft gun, Laser pointer and Hardware switch
    - $8: Amazon (1)
    - $1 (1) + $3: AliExpress.
5) PCA9685 motor driver module (1)
    - Used for I2C -> PWM to ctrl Servos
    - $12: Amazon (2)
6) X Servo (3 Wire, 20KG Servo Motor) (1)
    - I strongly suggest you copy the servos I have chosen. Makes life 100% easier when building.
    - Full disclosure, servo torque is arbitrary. I guessed... May be overkill.
    - $15: Amazon (1)
7) Y Servo (3 Wire, 35KG Servo Motor) (1)
    - I strongly suggest you copy the servos I have chosen. Makes life 100% easier when building.
    - Full disclosure, servo torque is arbitrary. I guessed... May be overkill.
    - $28: Amazon (1)
8) USB Laser pointer (1)
    - Cheapest laser pointer that has USB charging.
    - $10: Amazon (1)
9) Airsoft gun (1)
    - Any airsoft gun works however, if you pick an 'AEG' (Electric) the theory behind the firing mechanism will still work.
    - I used "Lancer Tactical Gen 2 LT-19 airsoft M4 Carbine 10" because its what I had lying around. Documentation and CAD files match this.
    - $189: Amazon (1)
10) 12V Battery (A ___AH is required for 4 hours of on-time) (1)
    - I currently am using a 10AH battery.
    - Load calculations will determine the final size of the battery.
    - $40: Amazon (1)
11) 12V -> 7.6V DC-DC Buck converter (1)
    - Used to step from 12 -> 7.6 for Servos
    - I used Amazons Generic: 'DC-DC Buck Converter 3.5-30V to 0.8-29V 10A....'
    - $13: Amazon (1)
12) 12V DC jack 2.1mm ID/ 5.5mm OD (2)
    - Used to power Arduino
    - $7: Amazon (5)
13) Rocker Switch 0N/OFF (1)
    - Used as Hardware safety for airsoft gun and servos
    - $8: Amazon (10)
14) NRF24L01+ Wireless Transceiver Module (1)
    - Radio communication between wrangler/turrent
    - $8: Amazon (4)
15) 3D Printer Filament
    - For turret 10KG of fillament, 
    - For wrangler 1KG of fillament,
    - I purchased bulk PLA fillament from AliExpress 10KG for 83$.
    - If Purchased off Amazon: $13 e/a
    - AliExpress:
      - Turret: $83
      - Wrangler: $9
    - Amazon:
      - Turrent: $130
      - Wrangler: $13
16) 4 Position Joystick (1)
    - Used to control wrangler
    - $15: Amazon (1)
17) Momentary Push button (1)
    - Used for Firing Wrangler
    - $11: Amazon (24)
18) USB camera (1)
    - Used for image detection
    - $20: Amazon(1)
19) Assorted items
    - Wire
      - 14,16,18 and regular breadboard wires were used.
      - I used higher quality 18 gauge solid copper wires for logic wires to help keep organized.
    - Dupont connections (OPT)
      - I used this sometimes to terminate my 18 gauge wire when it was being used...
    - Solder + Iron
    - Assorted circuitry components...
      - (2) 10uF Capacitors
      - (1) 10K Resistor
    - 9v Battery to power Wrangler Arduino

## Total:
Price Depends on where you purchase from:
- Amazon:
    - $576 
- Amazon/AliExpress:
    - $498

This price is quite steep. A major price cut could be made by choosing a different Airsoft gun, $189 price is steep.

I have had most pieces lying around. I spent maybe 300$ to get this all going.