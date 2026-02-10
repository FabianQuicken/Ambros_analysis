import numpy as np
import math

trajectories = [
    {
        "start_frame": 10234,
        "end_frame": 11890,
        "duration_s": 55.3
    }
]


dic = {
        "start_frame": 12234,
        "end_frame": 13890,
        "duration_s": 40.3
    }

trajectories.append(dic)




trajectories2 = [
    {
        "start_frame": 14234,
        "end_frame": 16890,
        "duration_s": 455.3
    },
    {
        "start_frame": 18234,
        "end_frame": 20890,
        "duration_s": 155.3
    }
]

trajectories = trajectories + trajectories2


a = [-1, -5, -10]
#print(min(a))


a = (1, 1)
b1 = (6, 1)
b2 = (6, 6)

# punkte als array für einfachere rechnungen
a = np.asarray(a, float)
b1 = np.asarray(b1, float)
b2 = np.asarray(b2, float)

# vektoren berechnen ("spitze minus fuss")
v1 = b1 - a
v2 = b2 - a

# wir wollen den inneren Winkel θ Tetha berechnen, der von beiden Vektoren eingeschlossen wird
# cos θ = Skalarprodukt v1 * v2 dividied by |v1| * |v2|
# daraus folgt θ = cos^-1 ((Skalarprodukt v1 * v2) / (|v1| * |v2|))

# skalarprodukt v1 * v2:
sp = v1[0] * v2[0] + v1[1] * v2[1]

# betrag v1 und v2:
betrag1 = math.sqrt(v1[0]**2 + v1[1]**2)
betrag2 = math.sqrt(v2[0]**2 + v2[1]**2)

# tetha berechnen
radians = np.arccos(sp / (betrag1 * betrag2))
degrees = np.degrees(radians)
print(degrees)

a = {'mouse1 and mouse2': [], 'mouse1 and mouse3': [], 'mouse2 and mouse3': [(np.int64(1907), np.int64(1937))]}

for entry in a:
    print(entry)