# Constants for normalization ranges of dynamic variables
DYNAMIC_CONTEXT_MIN_MAX = {
    'time_second': (0.0, 47695.0),
    'temperature_out': (-8.5, 37.5),
    'steering_speed': (0.0, 782.5419211881665),
    'avg_irradiation': (0.0, 1000.0),
    'KBI_speed': (0.0, 249.2525111761982),
    'soc': (0.0, 98.9),
    'latitude': (0.0, 52.408768),
    'longitude': (0.0, 13.634248),
    'altitude': (-241.2428200611689, 1446.0),
    'rain_sensor': (0.0, 100.0)
}

# Mappings for static variables
WEEKDAY = {
    'weekday2.0': 0, 'weekday4.0': 1, 'weekday5.0': 2, 'weekday6.0': 3,
    'weekday0.0': 4, 'weekday1.0': 5, 'weekday3.0': 6
}
HOUR = {
    'hour21.0': 7, 'hour17.0': 8, 'hour18.0': 9, 'hour10.0': 10, 'hour12.0': 11, 'hour7.0': 12, 'hour8.0': 13,
    'hour9.0': 14, 'hour11.0': 15, 'hour15.0': 16, 'hour19.0': 17, 'hour16.0': 18, 'hour22.0': 19, 'hour13.0': 20,
    'hour20.0': 21, 'hour14.0': 22, 'hour1.0': 23, 'hour6.0': 24, 'hour23.0': 25, 'hour2.0': 26, 'hour3.0': 27,
    'hour0.0': 28, 'hour4.0': 29, 'hour5.0': 30
}
MONTH = {
    'month9.0': 31, 'month12.0': 32, 'month2.0': 33, 'month4.0': 34, 'month5.0': 35, 'month6.0': 36, 'month7.0': 37,
    'month10.0': 38, 'month11.0': 39, 'month1.0': 40, 'month3.0': 41
}
KICKDOWN = {'kickdown0.0': 42, 'kickdown1.0': 43}
SEATBELT_CODRIVER = {'seatbelt_codriver1.0': 44, 'seatbelt_codriver0.0': 45}
SEATBELT_REAR_R = {'seatbelt_rear_r0.0': 46, 'seatbelt_rear_r1.0': 47}
SEATBELT_REAR_L = {'seatbelt_rear_l1.0': 48, 'seatbelt_rear_l0.0': 49}
SEATBELT_REAR_M = {'seatbelt_rear_m0.0': 50, 'seatbelt_rear_m1.0': 51, 'seatbelt_rear_mnicht_verbaut': 52}

# Mapping for last interactions, distinguished from other static variables
LAST_INTERACTION = {
    'last_interactionmedia/selectedSource/Bluetooth': 53,
    'last_interactionmedia/selectedSource/Radio': 54,
    'last_interactionmedia/selectedSource/Favorite': 55,
    'last_interactionnavi/Start/Favorite': 56,
    'last_interactionnavi/Start/Address': 57,
    'last_interactionphone/goTo/Favorite': 58,
    'last_interactioncar/driveMode/0.0': 59,
    'last_interactioncar/driveMode/2.0': 60,
    'last_interactionphone/Start/CarPlay': 61,
    'last_interactionclima/AC/on': 62,
    'last_interactioncar/driveMode/1.0': 63,
    'last_interactioncar/charismaLevel/Lift': 64,
    'last_interactioncar/charismaLevel/Abgesenkt': 65,
    'last_interactioncar/ESS/on': 66,
    'last_interactioncar/driveMode/3.0': 67,
    'last_interactionphone/Start/AndroidAuto': 68,
    'last_interactionclima/AC/off': 69,
    'last_interactioncar/driveMode/0': 70,
    'last_interactioncar/charismaLevel/Tief': 71,
    'last_interactioncar/driveMode/3': 72,
    'last_interactioncar/driveMode/2': 73,
    'last_interactionphone/Connect/NewDevice': 74,
    'last_interactionphone/Call/Favorite': 75,
    'last_interactionphone/Call/PersonX': 76,
    'last_interactionclima/AC/ECO': 77,
    'last_interactioncar/charismaLevel/Mittel': 78,
    'last_interactioncar/Start/ParkAssistant': 79,
    'last_interactionmedia/selectedSource/CarPlay': 80
}

previous_interactions = 0
# Check for previous interactions and adjust mapping accordingly
if not previous_interactions:
    last_interaction = 10

# Mapping for output labels or targets
LABELS = {
    'clima/AC/on': 0, 'media/selectedSource/Radio': 1, 'car/charismaLevel/Tief': 2, 'phone/Connect/NewDevice': 3,
    'clima/AC/ECO': 4, 'car/charismaLevel/Mittel': 5, 'car/driveMode/3.0': 6, 'navi/Start/Favorite': 7,
    'car/Start/ParkAssistant': 8, 'phone/goTo/Favorite': 9, 'car/driveMode/3': 10, 'phone/Call/PersonX': 11,
    'car/charismaLevel/Lift': 12, 'car/ESS/on': 13, 'car/driveMode/1.0': 14, 'navi/Start/Address': 15, 
    'car/driveMode/2.0': 16, 'phone/Call/Favorite': 17, 'car/driveMode/2': 18, 'media/selectedSource/Bluetooth': 19,
    'car/charismaLevel/Abgesenkt': 20, 'media/selectedSource/Favorite': 21, 'clima/AC/off': 22,
    'media/selectedSource/CarPlay': 23, 'phone/Start/AndroidAuto': 24, 'phone/Start/CarPlay': 25,
    'car/driveMode/0.0': 26, 'car/driveMode/0': 27
}
