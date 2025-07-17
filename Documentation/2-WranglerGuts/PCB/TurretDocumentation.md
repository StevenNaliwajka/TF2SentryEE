PCB fabrication

Fast charging of 12V Lead Acid battery.
- [CN3722 Datasheet](https://www.alldatasheet.com/datasheet-pdf/pdf/1132128/CONSONANCE/CN3722.html)


USB version is differnt than PD version.
USB w/ PD 3.1 is the 'biggest power delivery' 
- 48V 5A through USB-C

Drop it down to an acceptible charging current

LeadACID battery charging current is safe @ .3(C) charging rate
where C = Amp Hours of the battery.

How to know the AH of the battery?
If given RC (Reserve Capacity) - How long the battery can cary a 25A amp draw before dropping to 10.5V.

AH = RC(min) * 25/ 60
taking the RC of a battery off [walmart](https://duckduckgo.com/y.js?ad_domain=walmart.com&ad_provider=bing&ad_type=prad&click_metadata=Vj%2DTnmWMVkpqa8CMfkEGyP_gqpLnYzvwR3do_x_dsBA.NptQoPc6AtcbhRZXlVoGBA&eddgt=6T6Dw2P2OkQ20UwlGlnIhg%3D%3D&rut=c9a1fdaea8bc70221052e425fcad899c4f16283c99dbbb8ac0898391ee1c5a46&u3=https%3A%2F%2Fwww.bing.com%2Faclick%3Fld%3De81V_usR8fMsreBme56GOKijVUCUxBCnvvJVay2Y4gA3DpL%2DY58%2DPWoUbpIhMmWYKB6r4dCUMOesJsve8Fk4ajiPY9hs9TXM0Sy3oTFgkA78ND6EcTjDO5ASx9AVEM4rvF2CHYB8iF6OLQb0dzN2Y2Hkg6WTEJEP6y5hlR1nDnhVaIZ2VSvzCvHrNjdXwW65sUmciWNoQuAWdPCLcz%2DT1aCqot4IJR8GiVJE59axzHU7gp8_NwnNute6hAo_SviDfWyVJuyaJKuYp4H8KGiOCeusU7aj3DYwlMeiLKisNFIZfZfAwLBXdUfmgMt1glQelqZpbXeJCmxjZ5q2r8WrkvsP4MA17igNKmgE3P4cwXekS7tQE%2DxI7GEzqEYgHuQF85TvrsIo92nH7ErkT320Vw7MyXPBLvBKr6CF0_4SRpL_0c45YV7KsBbwD7nn%2DQOw64zL4kJe8V6NPseJLVzbe7zvqPM5JZWkIeAnB8uH3miDOH8ok0Z6mIA5fWsw8k9QraOTdCeMag6fGKCFU7G9ZfaSV9TMNOXsAIPOko4LZG8spjBb3ys%2DuBUmhBGHgrm5KNEfIhNMu8go7cfg4JKIfWav9a5VLEsHneYacw_I1jU9EtdnWbxQVECC9bwxc_oluhkwzGpp6TcPtMmLzMmQv782NW5L0bNGlR24lKWu3JYbxWAQYGIMbqHOYhKlfNM_Ipsr_j47YVx0x204D0iniaRGC3l6MEDLOs_V_HxZpbIbUB7rhuN1DEc9QMzgDiu%2D1FoxPRyw%26u%3DaHR0cHMlM2ElMmYlMmZjbGlja3NlcnZlLmRhcnRzZWFyY2gubmV0JTJmbGluayUyZmNsaWNrJTNmJTI2JTI2ZHNfZV9hZGlkJTNkNzI1NjgxMDQ4MjQxNTQlMjZkc19lX3RhcmdldF9pZCUzZHBsYS00NTc2MTY3NDIwNTA1OTQwJTI2ZHNfZV9wcm9kdWN0X2dyb3VwX2lkJTNkNDU3NjE2NzQyMDUwNTk0MCUyNmRzX2VfcHJvZHVjdF9pZCUzZDI4Mjc1NjcyXzAlMjZkc19lX3Byb2R1Y3RfY291bnRyeSUzZFVTJTI2ZHNfZV9wcm9kdWN0X2xhbmd1YWdlJTNkRU4lMjZkc19lX3Byb2R1Y3RfY2hhbm5lbCUzZExvY2FsJTI2ZHNfdXJsX3YlM2QyJTI2ZHNfZGVzdF91cmwlM2RodHRwcyUzYSUyZiUyZnd3dy53YWxtYXJ0LmNvbSUyZmlwJTJmRXZlclN0YXJ0LVBsdXMtTGVhZC1BY2lkLUF1dG9tb3RpdmUtQmF0dGVyeS1Hcm91cC03OC0xMi1Wb2x0LTYwMC1DQ0ElMmYyODI3NTY3MiUzZndtbHNwYXJ0bmVyJTNkd2xwYSUyNnNlbGVjdGVkU2VsbGVySWQlM2QwJTI2d2wxMyUzZDM0MDElMjZhZGlkJTNkMjIyMjIyMjIyMzI2MDAxMzY1OTEwXzExNjEwODYxMjI2MzY1MzRfbGlhJTI2d21sc3BhcnRuZXIlM2R3bXRsYWJzJTI2d2wwJTNkZSUyNndsMSUzZG8lMjZ3bDIlM2RjJTI2d2wzJTNkNzI1NjgxMDQ4MjQxNTQlMjZ3bDQlM2RwbGEtNDU3NjE2NzQyMDUwNTk0MCUyNndsNSUzZDc4NTAwJTI2d2w2JTNkJTI2d2w3JTNkJTI2d2wxMCUzZFdhbG1hcnQlMjZ3bDExJTNkTG9jYWwlMjZ3bDEyJTNkMjgyNzU2NzJfMCUyNndsMTQlM2R3YWxtYXJ0JTI1MjBjYXIlMjUyMGJhdHRlcnklMjZ2ZWglM2RzZW0lMjZtc2Nsa2lkJTNkYjEzZmJkMjQyMzVjMWM0OTFlNWRkNTQ5ODFlMDA4NDQ%26rlid%3Db13fbd24235c1c491e5dd54981e00844&vqd=4-169645248779270366465273386037255701107&iurl=%7B1%7DIG%3D61F91D22D8AA4AE2B434C79895B901B0%26CID%3D1166E4FEC41061D71F76F2DDC5896080%26ID%3DDevEx%2C5076.1)
It can sustain for 120 minutes.

so the ah (120 * 25) / 60 = 50 AH

TODO:
GET spec for CN3722
Should I use a battery not designed for CARS?
Would that give me more AH?
What is a 'general' way to go about the batteries to meet all edge cases.

POWER:
- USB3.0 w/ PD3 or PD3.1
  - Determine if that much power is worth it.
  - Verify that it is 'backwards compatable'
    - Usb 2.0 or Usb w/o PD will still allow the equipment to function.
- Convert power to 28 volts.
  - IF PD3. NO need for this, CN3722 will take the 20V.
  - If PD3.1, Need voltge drop from 48V to X?
- Charge the car battery w/ CN3722. 
  - Utilize CCCA (Coulomb Counting + Current Control Algorithm)
    - Try an implement in hardware. Worst case, link to the Main controller (ft232h)
  - Figure out max current allowed. 
    - I dont know the size of battery.
    - I dont know the AH.
    - Assume its a Lead Acid car battery still.
    - Want to get the max throughput and take advantage of supported.

BRAIN:
- Assume using FT232H for the Brain.
- Use SPI for communication protocol:
  - SPI will allow the elimination of the aurduino chip from the circuit
  - Motor Controller + nLRF24L01 will both use SPI

RADIO: 
- Assume nLRF24L01 + helper chip.
- However, Run it through SPI. 
  - I used I2C before, verify that it will work properly.

Motor Controller:
- Find an acceptable Motor controller to control two (2) PWM managed servo motors.

Switching:
- Use mosfets to switch power quickly and effectively.
- 