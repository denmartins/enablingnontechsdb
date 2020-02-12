from models.semioticmodule import *
import datamanagement.dataaccess as dt
from models.customer import *
from models.clause import *
from models.car import Car
from models.mcda import Criterion

original_cars, cars, preprocessed_cars = dt.DAO.get_cars()

def get_cars():
    return original_cars, cars, preprocessed_cars

def get_customers_view():
    customer_1 = Customer()
    customer_1.preferences = [DiadicClause(Car.TYPE, Operation.EQUALS, 'Sporty'), 
                            DiadicClause(Car.PRICE, Operation.LESS_THAN_EQUALS, 7000),
                            DiadicClause(Car.MANUFACTURER, Operation.EQUALS, 'Ford')]

    customer_1.profile = {Customer.AGE: 25, 
                        Customer.CIVIL_STATE: Customer.SINGLE_CIVIL_STATE, 
                        Customer.SEX: Customer.MALE_SEX, Customer.KIDS: 0}

    customer_1.criteria = [Criterion(Car.PRICE, maximize=False, weight=0.6), 
                       Criterion(Car.HORSEPOWER, maximize=True, weight=0.4)]

    customer_2 = Customer()
    customer_2.preferences = [DiadicClause(Car.PASSENGER_CAPACITY, Operation.GREATER_THAN_EQUALS, 5),
                        DiadicClause(Car.PRICE, Operation.LESS_THAN_EQUALS, 9000),
                        DiadicClause(Car.LUGGAGE_CAPACITY, Operation.GREATER_THAN_EQUALS, 20)]

    customer_2.profile = {Customer.AGE: 32,
                    Customer.CIVIL_STATE: Customer.SINGLE_CIVIL_STATE,
                    Customer.SEX: Customer.FEMALE_SEX,
                    Customer.KIDS: 2}

    customer_2.criteria = [Criterion(Car.PRICE, maximize=False, weight=0.6),
                        Criterion(Car.LUGGAGE_CAPACITY, maximize=True, weight=0.4)]

    customer_3 = Customer()
    customer_3.preferences = [DiadicClause(Car.AUTOMATIC_GEARBOX, Operation.EQUALS, 1),
                          DiadicClause(Car.PRICE, Operation.LESS_THAN_EQUALS, 7500),
                          DiadicClause(Car.MPG, Operation.GREATER_THAN_EQUALS, 25)]

    customer_3.criteria = [Criterion(Car.HORSEPOWER, maximize=True, weight=0.3),
                       Criterion(Car.MPG, maximize=True, weight=0.7)]

    customers = [customer_1, customer_2, customer_3]

    return customers

def get_customers():
    customer_1 = Customer()
    customer_1.preferences = [DiadicClause(Car.TYPE, Operation.EQUALS, 'Sporty'), 
                            DiadicClause(Car.PRICE, Operation.LESS_THAN_EQUALS, 
                                         dt.DAO.normalizers[Car.PRICE].get_normalized(7000)),
                            DiadicClause(Car.MANUFACTURER, Operation.EQUALS, 'Ford')]

    customer_1.profile = {Customer.AGE: 25, 
                        Customer.CIVIL_STATE: Customer.SINGLE_CIVIL_STATE, 
                        Customer.SEX: Customer.MALE_SEX, Customer.KIDS: 0}

    customer_1.criteria = [Criterion(Car.PRICE, maximize=False, weight=0.6), 
                       Criterion(Car.HORSEPOWER, maximize=True, weight=0.4)]

    customer_2 = Customer()
    customer_2.preferences = [DiadicClause(Car.PASSENGER_CAPACITY, Operation.GREATER_THAN_EQUALS, 
                                     dt.DAO.normalizers[Car.PASSENGER_CAPACITY].get_normalized(5)),
                        DiadicClause(Car.PRICE, Operation.LESS_THAN_EQUALS, 
                                     dt.DAO.normalizers[Car.PRICE].get_normalized(9000)),
                        DiadicClause(Car.LUGGAGE_CAPACITY, Operation.GREATER_THAN_EQUALS, 
                                     dt.DAO.normalizers[Car.LUGGAGE_CAPACITY].get_normalized(20))]

    customer_2.profile = {Customer.AGE: 32,
                    Customer.CIVIL_STATE: Customer.SINGLE_CIVIL_STATE,
                    Customer.SEX: Customer.FEMALE_SEX,
                    Customer.KIDS: 2}

    customer_2.criteria = [Criterion(Car.PRICE, maximize=False, weight=0.6),
                        Criterion(Car.LUGGAGE_CAPACITY, maximize=True, weight=0.4)]

    customer_3 = Customer()
    customer_3.preferences = [DiadicClause(Car.AUTOMATIC_GEARBOX, Operation.EQUALS, 1),
                          DiadicClause(Car.PRICE, Operation.LESS_THAN_EQUALS, 
                                       dt.DAO.normalizers[Car.PRICE].get_normalized(7500)),
                          DiadicClause(Car.MPG, Operation.GREATER_THAN_EQUALS, 
                                       dt.DAO.normalizers[Car.MPG].get_normalized(25))]

    customer_3.criteria = [Criterion(Car.HORSEPOWER, maximize=True, weight=0.3),
                       Criterion(Car.MPG, maximize=True, weight=0.7)]

    customers = [customer_1, customer_2, customer_3]

    return customers

#def get_additional_preferences():
#    return [DiadicClause(Car.MPG, Operation.GREATER_THAN_EQUALS, 
#                         dt.DAO.normalizers[Car.MPG].get_normalized(20)),
#            DiadicClause(Car.HORSEPOWER, Operation.GREATER_THAN_EQUALS, 
#                         dt.DAO.normalizers[Car.HORSEPOWER].get_normalized(120)),
#            DiadicClause(Car.NUM_OF_CYLINDERS, Operation.EQUALS, 
#                         dt.DAO.normalizers[Car.NUM_OF_CYLINDERS].get_normalized(6))]

def get_additional_preferences():
    return [DiadicClause(Car.MPG, Operation.GREATER_THAN_EQUALS, 20),
            DiadicClause(Car.HORSEPOWER, Operation.GREATER_THAN_EQUALS, 120),
            DiadicClause(Car.NUM_OF_CYLINDERS, Operation.EQUALS, 6)]


def get_operations_dict():
    return {Operation.EQUALS: '=', Operation.GREATER_THAN: '>', 
            Operation.GREATER_THAN_EQUALS: '>=', 
            Operation.LESS_THAN: '<', Operation.LESS_THAN_EQUALS: '<='}