import serial
import time

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

def send_message(message: str):
    bluetooth.write(message.encode())  # Send message as bytes
    time.sleep(0.1)                    # Small delay to allow processing

if __name__ == "__main__":
    try:
        update_car(0)

        while True:
            try:
                test_model_output = int(input("Enter for function update_car: "))
            except ValueError:
                print("Please only enter a number between 0 and 8")
                continue
            update_car(test_model_output)

    finally:
        bluetooth.close()  # Ensure the Bluetooth connection is closed after use

