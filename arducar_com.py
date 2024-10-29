import serial
import time

# ------------------------------------------------------------
#
# DEFAULT VALUES - ONLY CHANGE THIS TO MATCH FROM THE MODEL OUTPUT. Ben, Joe, Kevin
#
# ------------------------------------------------------------

class_output = 0
action = "S"
speed = "0"

# ------------------------------------------------------------
#
# SWITCH CASES: Edward will Write this part.
#
# ------------------------------------------------------------



# ------------------------------------------------------------
#
# SENDING COMMANDS
#
# ------------------------------------------------------------

# Update 'COMX' with the port that corresponds to your HC-05 module
bluetooth = serial.Serial(port='COM7', baudrate=9600, timeout=1)

def update_car(model_output: int):
    # class output will translate to:
    action_lookup = [("F", "1"),
                     ("F", "1"),
                     ("F", "2"),
                     ("F", "3"),
                     ("F", "4"),
                     ("F", "5"),
                     ("F", "6"),
                     ("F", "7"),
                     ("S", "0")]
    
    try:
        action, speed = action_lookup[model_output]
    except IndexError:
        action, speed = "S", "0"
    send_message(action)
    send_message(speed)

def send_message(message):
    bluetooth.write(message.encode())  # Send message as bytes
    time.sleep(0.1)                    # Small delay to allow processing

if __name__ == "__main__":
    try:

        send_message(action)
        send_message(speed)

        update_car(-1)

        def live_command_test():
            while (True):
                command = input("Enter: ")
                send_message(command)

        #####live_command_test()

    finally:
        bluetooth.close()  # Ensure the Bluetooth connection is closed after use

