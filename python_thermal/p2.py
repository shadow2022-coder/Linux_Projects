import subprocess
import tkinter as tk
from tkinter import ttk

def get_thermal_data():
    """Get thermal data with zone names"""
    try:
        # Get thermal zone names
        result_names = subprocess.run(
            "ls /sys/class/thermal/thermal_zone* -d",
            shell=True,
            capture_output=True,
            text=True
        )
        
        # Get temperature values
        result_temps = subprocess.run(
            "cat /sys/class/thermal/thermal_zone*/temp",
            shell=True,
            capture_output=True,
            text=True
        )
        
        zones = result_names.stdout.strip().split('\n')
        temps = result_temps.stdout.strip().split('\n')
        
        data = []
        for zone, temp in zip(zones, temps):
            zone_name = zone.split('/')[-1]
            temp_celsius = int(temp) / 1000  # Convert from millidegrees
            data.append((zone_name, temp_celsius))
        
        return data
    except Exception as e:
        return [("Error", str(e))]

def display_gui():
    root = tk.Tk()
    root.title("Thermal Monitor")
    root.geometry("600x400")
    
    # Create Treeview (table view)
    tree = ttk.Treeview(root, columns=("Zone", "Temperature"), height=15)
    tree.heading("#0", text="Index")
    tree.heading("Zone", text="Thermal Zone")
    tree.heading("Temperature", text="Temperature (°C)")
    
    tree.column("#0", width=50)
    tree.column("Zone", width=250)
    tree.column("Temperature", width=200)
    
    tree.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    def populate_data():
        # Clear existing data
        for item in tree.get_children():
            tree.delete(item)
        
        # Insert new data
        data = get_thermal_data()
        for idx, (zone, temp) in enumerate(data, 1):
            tree.insert("", tk.END, text=str(idx), values=(zone, f"{temp:.2f}"))
    
    # Initial population
    populate_data()
    
    # Refresh button
    def refresh():
        populate_data()
    
    refresh_btn = tk.Button(root, text="Refresh", command=refresh)
    refresh_btn.pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    display_gui()