import pandas as pd
from models.car import *
from datamanagement.datanormalizer import *
from sklearn import preprocessing

class DataAccessObject:
    def __init__(self):
        self.normalizers = {}

    def get_car_attributes(self):
        return [Car.MAKE, Car.MANUFACTURER, Car.TYPE, Car.PRICE, Car.MPG, 
                Car.NUM_OF_CYLINDERS, Car.HORSEPOWER, Car.FUEL_TANK_CAPACITY, 
                Car.RPM, Car.WHEELBASE, Car.REAT_SEAT, Car.WEIGHT, Car.AUTOMATIC_GEARBOX, 
                Car.PASSENGER_CAPACITY, Car.LENGTH, Car.WIDTH, Car.LUGGAGE_CAPACITY,
                Car.AIR_BAGS, Car.DRIVE_TRAIN, Car.ORIGIN, Car.IMAGEPATH]

    def get_cars(self):
        car_columns = [Car.MANUFACTURER, Car.MODEL, Car.TYPE,"Min.Price", Car.PRICE,"Max.Price",
                   "MPG.city","MPG.highway",Car.AIR_BAGS, Car.DRIVE_TRAIN, Car.NUM_OF_CYLINDERS,
                   Car.ENGINE_SIZE, Car.HORSEPOWER,Car.RPM,Car.REV_PER_MILE,"manual_transmission",
                   Car.FUEL_TANK_CAPACITY, Car.PASSENGER_CAPACITY, Car.LENGTH, Car.WHEELBASE, Car.WIDTH,
                   Car.TURN_CIRCLE, Car.REAT_SEAT, Car.LUGGAGE_CAPACITY, Car.WEIGHT, Car.ORIGIN, Car.MAKE]

        # Load car data
        cars = pd.read_table('database\\original_dataset.txt', sep=',',names=car_columns, engine='python')

        # Calculate average fuel consumption
        cars[Car.MPG] = (cars["MPG.city"] + cars["MPG.highway"])/2
    
        # Update price. We consider only 50% of the price to simulate used cars price
        cars[Car.PRICE] = cars[Car.PRICE] * 1000 / 2
        
        # Fill NA with interpolation
        cars[Car.LUGGAGE_CAPACITY] = cars[Car.LUGGAGE_CAPACITY].interpolate(method='spline', order=1, limit=int(cars[Car.LUGGAGE_CAPACITY].max()), limit_direction='both')
        cars[Car.REAT_SEAT] = cars[Car.REAT_SEAT].interpolate(method='spline', order=1, limit=int(cars[Car.REAT_SEAT].max()), limit_direction='both')

        # Map manual transmition yes to no automatic gearbox
        mapping = {'Yes': 0, 'No': 1}
        cars[Car.AUTOMATIC_GEARBOX] = cars["manual_transmission"].map(mapping)

        # Map origin
        mapping = {'non-USA': 1, 'USA': 0}
        cars[Car.ORIGIN] = cars[Car.ORIGIN].map(mapping)

        cars[Car.IMAGEPATH] = ['01acura_integra.jpg', '02Acura_Legend.jpg', '03audi_100.jpg', 
                               '04audi_90.jpg', '05bmw_535i.jpg', '06buick_century.jpg', 
                               '07buick_lesabre.jpeg', '08buick_roadmaster.jpg', '09buick_riviera.jpg', 
                               '10Cadillac_DeVille.jpg', '11Cadillac_Seville.jpg', 
                               '12chevrolet_cavalier.jpg', '13Chevrolet_Corsica.jpg', 
                               '14chevrolet_camaro.jpg', '15Chevrolet_Lumina.jpg', 
                               '16Chevrolet_Lumina_APV.jpg', '17chevrolet_astro.jpg', 
                               '18Chevrolet_Caprice.jpg', '19chevrolet_corvette.jpeg', 
                               '20Chrysler_Concorde.jpg', '21Chrysler_lebaron.jpg', 
                               '22chrysler_imperial.jpg', '23dodge_colt.jpg', 
                               '24Dodge_Shadow.jpg', '25Dodge_Spirit.jpg', '26dodge_caravan.jpg', 
                               '27Dodge_Dynasty.jpg', '28dodge_stealth.jpg', '29Eagle_Summit.jpg', 
                               '30eagle_vision.jpg', '31Ford_Festiva.jpg', '32Ford_Escort.jpg', 
                               '33Ford_Tempo.jpg', '34ford_mustang.jpg', '35Ford_Probe.jpg', 
                               '36Ford_Aerostar.jpg', '37Ford_Taurus.jpg', '38Ford_Crown_Victoria.jpg', 
                               '39Geo_Metro.jpg', '40geo_storm.jpeg', '41honda_prelude.jpg', 
                               '42Honda_Civic.JPG', '43Honda_Accord.jpg', '44Hyundai_excel.jpg', 
                               '45Hyundai_Lantra.jpg', '46Hyundai_Scoupe.jpg', '47HYUNDAI_Sonata.jpg', 
                               '48Infiniti_Q45.jpg', '49Lexus_ES300.jpg', '50lexus_sc300.jpg', 
                               '51lincoln_continental.jpg', '52Lincoln_Town_Car.jpg', '53Mazda_323.jpeg', 
                               '54mazda_protege.jpg', '55mazda_626.jpg', '56Mazda_MPV.jpg', 
                               '57mazda_rx7.jpg', '58Mercedes_Benz_190E.JPG', '59Mercedes_Benz_300E.jpg', 
                               '60mercury_capri.jpg', '61Mercury_Cougar.jpg', '62mitsubishi_mirage.jpg', 
                               '63mitsubishi_diamante.jpg', '64nissan_sentra.jpg', '65Nissan_Altima.jpg', 
                               '66nissan_quest.jpg', '67nissan_maxima.jpg', '68oldsmobile_achieva.jpg', 
                               '69oldsmobile_cutlass_ciera.jpg', '70oldsmobile_silhouette.jpg', 
                               '71oldsmobile_eighty_eight.jpg', '72plymouth_laser.jpg', '73pontiac_lemans.jpg', 
                               '74pontiac_sunbird.jpg', '75Pontiac_Firebird.jpg', '76pontiac_grand_prix.jpg', 
                               '77pontiac_bonneville.jpg', '78Saab_900.jpg', '79Saturn_SL.jpg', 
                               '80subaru_justy.jpg', '81subaru_loyale.jpg', '82subaru_legacy.jpg', 
                               '83suzuki_swift.jpg', '84toyota_tercel.jpg', '85toyota_celica.jpg', 
                               '86toyota_camry.jpg', '87toyota_previa.jpg', '88Volkswagen_Fox.jpg', 
                               '89volkswagen_eurovan.jpg', '90volkswagen_passat.jpg', '91volkswagen_corrado.jpg', 
                               '92volvo_240.jpeg', '93volvo_850.jpg']

        # Create a view with a subset of features
        car_view = cars[self.get_car_attributes()]

        original_cars = car_view.copy(True)
        self.normalize_data(car_view)

        preprocessed_data = self.get_preprocessed_data(car_view)
        #preprocessed_data = pd.read_pickle('database//processed_cars.pkl')

        return original_cars, car_view, preprocessed_data

    def normalize_data(self, cars):
        for col in [Car.PRICE, Car.HORSEPOWER, Car.NUM_OF_CYLINDERS, Car.PASSENGER_CAPACITY, 
                    Car.LUGGAGE_CAPACITY, Car.MPG, Car.FUEL_TANK_CAPACITY, Car.RPM, 
                    Car.WHEELBASE, Car.REAT_SEAT, Car.WEIGHT, Car.LENGTH, Car.WIDTH]:
            x = cars[col].values
            normalizer = ZeroOneNormalizer()
            norm_values = normalizer.normalize(x.reshape(-1, 1))
            self.normalizers[col] = normalizer
            cars[col] = pd.DataFrame(data=norm_values, index=range(1, len(norm_values) + 1))

    def get_preprocessed_data(self, data):
        car_data = data.copy(True)
        del car_data[Car.MAKE]
        del car_data[Car.IMAGEPATH]

        for col in [Car.MANUFACTURER, Car.TYPE, Car.AIR_BAGS, Car.DRIVE_TRAIN]:
            le = preprocessing.LabelEncoder()
            le.fit(car_data[col].values)
            transformed = le.transform(car_data[col].values)
            car_data[col] = pd.DataFrame(data=transformed, index=range(1, len(transformed) + 1))

            x = car_data[col].values
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(x.reshape(-1, 1))
            car_data[col] = pd.DataFrame(data=x_scaled, index=range(1, len(x_scaled) + 1))

        return car_data

DAO = DataAccessObject()