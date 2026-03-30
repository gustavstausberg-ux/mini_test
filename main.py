import math

# Parameter
v0 = 10  # Anfangsgeschwindigkeit (m/s)
angle_deg = 45  # Winkel in Grad
g = 9.81  # Erdbeschleunigung

# Umrechnung in Radiant
angle_rad = math.radians(angle_deg)

# Zeit bis Boden
t_flight = (2 * v0 * math.sin(angle_rad)) / g

# Reichweite
range_ = v0 * math.cos(angle_rad) * t_flight

print(f"Flight time: {t_flight:.2f} s")
print(f"Range: {range_:.2f} m")