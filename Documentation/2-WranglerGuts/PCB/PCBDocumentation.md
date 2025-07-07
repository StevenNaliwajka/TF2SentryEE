Lots of references here:

5V to 3.3V regulator:
- AMS1117 is whats used in the standard "nRF24L01+PA+LNA" 5->3.3V dedicated supply.
- [AMS1117 DOCS](http://www.advanced-monolithic.com/pdf/ds1117.pdf)
- [AMS1117 EX Circuit](https://theorycircuit.com/power-circuits/voltage-regulator-circuit-5v-to-3-3v/)  

BATTERY -> 5V Regulator:
 - [MT3608 DOCS](https://www.olimex.com/Products/Breadboarding/BB-PWR-3608/resources/MT3608.pdf)
 - [MT3608 EX Circuit](https://electronics.stackexchange.com/questions/477651/mt3608-improper-output-voltage)
 - [MT3608 Fixed 5V Thread](https://www.reddit.com/r/AskElectronics/comments/1fnrpkk/mt3608_fixed_5v_output/)
 - [Selecting an Inductor DOCS](https://www.monolithicpower.com/media/document/AN122%20Selecting%20a%20Boost%20Regulator%20and%20Its%20Inductor.pdf)
 - 

Battery MGMT Circuit + Load Sharing:
- [load sharing support thread](https://forum.arduino.cc/t/solar-li-ion-charging-with-power-path-load-sharing/602451)
- [load sharing thread #2](https://electronics.stackexchange.com/questions/591555/tp4056-with-load-sharing)
- [formal load sharing writeup](https://www.thanassis.space/loadsharing.html)
- [formal load sharing writeup #2](https://blog.zakkemble.net/a-lithium-battery-charger-with-load-sharing/)
- [load sharing thread #3](https://forum.arduino.cc/t/tp4056-lithium-charger-module-modification-to-add-power-sharing/1008473)
- [TP4056 load sharing writeup #3](https://www.best-microcontroller-projects.com/tp4056-page2.html)
- [TP4056 load sharing thread #4](https://www.reddit.com/r/PrintedCircuitBoard/comments/1k89rtd/tp4056_modul_and_load_charing_circuit/)
- [TP4056 load sharing thread #5](https://electronics.stackexchange.com/questions/628664/do-these-tp4056-charging-boards-have-built-in-load-sharing)
- [TP4056 voltage output thread](https://www.reddit.com/r/AskElectronics/comments/gyblxk/what_voltage_can_the_tp4056_output/)

nLRF24L01 Radio Circuit:
- [Example Circuit](https://imgur.com/a/0xGPQbj)
- [nLRF24L01 DOCS](https://cdn.sparkfun.com/assets/3/d/8/5/1/nRF24L01P_Product_Specification_1_0.pdf)
- USE RP-SMA connector?
- [Breakout nLRF24L01](https://www.amazon.com/UMLIFE-NRF24L01-Transceiver-Wireless-Regulator/dp/B096DLGV8F)
- [Logic LEVEL shifting Thread](https://forum.arduino.cc/t/nrf24l01-pa-lna-level-shifting/1052803)
- 

ATMEGA32u4 + USBC for flashing.
- [Minimal Circuit](https://image.easyeda.com/components/396fc048a63048f4991b64bf148cbc53.png)
- [USB-C + ATmega32u4 thread](https://electronics.stackexchange.com/questions/624732/connecting-usb-c-port-with-atmega32u4-questions)
- [USB + ATmega32u4 thread #2](https://forum.arduino.cc/t/atmega32u4-usb-powered-minimal-bootloader-and-test-circuit/437689)
- [USB-C Pinout](https://www.wandkey.com/wp-content/uploads/2024/01/usb-c-pinout-diagram.png)
- [Atmega32U4RC Bootloader Thread](https://forum.arduino.cc/t/does-atmega32u4rc-au-ic-come-with-bootloader/1069773/25)
- [ATmega32u4 Base Circuit Thread](https://forum.arduino.cc/t/atmega32u4-basic-circuit-power-supply-question/543325)
- [ATmega34U4 DOCS](https://ww1.microchip.com/downloads/en/devicedoc/atmel-7766-8-bit-avr-atmega16u4-32u4_datasheet.pdf)
- [ATmega34u4 BareBones Circuit](https://forum.arduino.cc/t/atmels-atmega-32u4/158473/2)
- [USB-C Pulldown resistors](https://www.reddit.com/r/AskElectronics/comments/18svx8z/usb_typec_receptacle_can_i_connect_both_cc1_and/)
- 