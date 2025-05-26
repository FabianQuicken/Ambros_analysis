from shapely.geometry import Point, Polygon



rectangle = [(50,-100), (1900,-100), (1900,-1050), (50,-1050)]

point = (50.1,-101)

print(Polygon(rectangle).contains(Point(point)))