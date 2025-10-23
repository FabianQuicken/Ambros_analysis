# hier entsteht der analyseteil für ein Multi Animal Experiment

# identity = False, d.h. es wird keine individuelle Maus analysiert, sondern die aufgenommene Kohorte als ganzes

# für identity = True, e.g. balb/c Maus + c57/bl6 Maus vermutlich eigene main.py später

# zur struktur: da die Multianimal dateien fehler beinhalten (id switches, fehlende daten, id overlays usw.), sollte die Analyse mit preprocessed Daten passieren, 
# nicht mit den raw dlc output files

# für non-social metrics macht es vermutlich sinn, über die Mäuse zu iterieren und die Daten am Ende zusammenzufassen - so könnte die single animal pipeline hier genutzt werden

# neue social metrics (e.g. social interaction) könnte separat analysiert werden

# nach Möglichkeit sollte die Pipeline mit verschieden vielen Individuen funktionieren


