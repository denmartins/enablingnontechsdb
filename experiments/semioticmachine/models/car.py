﻿class Car:
    TYPE = 'type'
    MANUFACTURER = 'manufacturer'
    HORSEPOWER = 'horsepower'
    PRICE = 'price'
    MPG = 'mpg'
    ENGINE_SIZE = 'engine_size'
    PASSENGER_CAPACITY = 'passenger_capacity'
    LENGTH = 'length'
    WIDTH = 'width'
    AUTOMATIC_GEARBOX = 'automatic_gearbox'
    LUGGAGE_CAPACITY = 'luggage_capacity'
    NUM_OF_CYLINDERS = 'num_of_cylinders'
    MODEL = 'model'
    MAKE = 'make'
    FUEL_TANK_CAPACITY = "fuel_tank_capacity"
    RPM = "RPM"
    WHEELBASE = "Wheelbase"
    REAT_SEAT = "Rear.seat.room"
    WEIGHT = "Weight"
    IMAGEPATH = 'imagepath'
    REV_PER_MILE = 'Rev.per.mile'
    TURN_CIRCLE = 'Turn.circle'
    AIR_BAGS = 'AirBags'
    DRIVE_TRAIN = 'DriveTrain'
    ORIGIN = 'Origin'

    def __init__(self, data):
        self.type = data[Car.TYPE]
        self.manufacturer = data[Car.MANUFACTURER]
        self.horsepower = data[Car.HORSEPOWER]
        self.price = data[Car.PRICE]
        self.fuel_consumption = data[Car.MPG]
        self.passenger_capacity = data[Car.PASSENGER_CAPACITY]
        self.length = data[Car.LENGTH]
        self.width = data[Car.WIDTH]
        self.automatic_gearbox = data[Car.AUTOMATIC_GEARBOX]
        self.luggage_capacity = data[Car.LUGGAGE_CAPACITY]
        self.num_of_cylinders = data[Car.NUM_OF_CYLINDERS]
        self.fuel_tank_capacity = data[Car.FUEL_TANK_CAPACITY]
        self.make = data[Car.MAKE]
        self.imagepath = data[Car.IMAGEPATH]