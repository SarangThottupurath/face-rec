import tkinter as tk
from datetime import datetime

class AgeCalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Age Calculator")

        # Create and place the widgets
        self.label1 = tk.Label(root, text="Enter your birthdate (YYYY-MM-DD):", font=('Arial', 14))
        self.label1.grid(row=0, column=0, pady=10)

        self.entry_birthdate = tk.Entry(root, font=('Arial', 14), width=20)
        self.entry_birthdate.grid(row=0, column=1, pady=10)

        self.calculate_button = tk.Button(root, text="Calculate Age", font=('Arial', 14), command=self.calculate_age)
        self.calculate_button.grid(row=1, column=0, columnspan=2, pady=10)

        self.result_label = tk.Label(root, text="Your Age: ", font=('Arial', 16))
        self.result_label.grid(row=2, column=0, columnspan=2, pady=10)

    def calculate_age(self):
        """Calculates and displays the age based on the input birthdate."""
        birthdate_str = self.entry_birthdate.get()
        
        try:
            # Parse the birthdate from the input
            birthdate = datetime.strptime(birthdate_str, "%Y-%m-%d")
            
            # Get today's date
            today = datetime.today()

            # Calculate the age by subtracting birth year from the current year
            age = today.year - birthdate.year
            
            # Adjust if the birthday hasn't occurred yet this year
            if (today.month, today.day) < (birthdate.month, birthdate.day):
                age -= 1

            # Display the calculated age
            self.result_label.config(text=f"Your Age: {age} years")
        
        except ValueError:
            self.result_label.config(text="Invalid date format! Please use YYYY-MM-DD.")

# Create the main window
root = tk.Tk()

# Create an instance of the AgeCalculatorApp
app = AgeCalculatorApp(root)

# Run the application
root.mainloop()
