import subprocess
import tkinter as tk
from tkinter import ttk

def get_thermal_data():
    """Execute the thermal command and return parsed data"""
    try:
        result = subprocess.run(
            "cat /sys/class/thermal/thermal_zone*/temp",
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr}"

def display_gui():
    """Display thermal data in a GUI window"""
    root = tk.Tk()
    root.title("Linux Thermal Monitor")
    root.geometry("500x400")
    
    # Add title
    title = tk.Label(root, text="Thermal Zone Temperatures", font=("Arial", 14, "bold"))
    title.pack(pady=10)
    
    # Add text widget to display data
    text_widget = tk.Text(root, height=15, width=50, font=("Courier", 10))
    text_widget.pack(padx=10, pady=10)
    
    # Get and display thermal data
    thermal_data = get_thermal_data()
    text_widget.insert("1.0", thermal_data)
    text_widget.config(state=tk.DISABLED)  # Make read-only
    
    # Add refresh button
    def refresh():
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        thermal_data = get_thermal_data()
        text_widget.insert("1.0", thermal_data)
        text_widget.config(state=tk.DISABLED)
    
    refresh_btn = tk.Button(root, text="Refresh", command=refresh)
    refresh_btn.pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    display_gui()