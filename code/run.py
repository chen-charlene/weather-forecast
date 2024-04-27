# import numpy as np
# data = np.load('../data/train_data/2015-01-01_00.npy')
# print(data.shape)

# import apache_beam   # Needs to be imported separately to avoid TypingError
import math
import numpy as np
from sklearn.model_selection import train_test_split
import weatherbench2
import xarray as xr
from weatherbench2 import config
from weatherbench2.regions import SliceRegion, ExtraTropicalRegion
from weatherbench2.metrics import MSE, ACC
from weatherbench2.evaluation import evaluate_in_memory

selection = {
    'variables': [
        'geopotential',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'vertical_velocity',
        '10m_wind_speed',
        'total_precipitation_6hr',
        'total_cloud_cover',
        '2m_temperature',
        'specific_humidity',
        'surface_pressure',
        'toa_incident_solar_radiation',
        'total_column_water_vapour'
    ],
    "levels": [500, 700, 850],
    # "time_slice": slice('2020-01-01', '2020-12-31'),
    "lat_slice": slice(30,50),
    "long_slice": slice(70,90),
}




def preprocess_data(split_percentage: float, window_size=10):
    obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'
    data = xr.open_zarr(obs_path)

    print("dataset shape: ", data.sizes)

    # data = data[selection['variables']].sel(level=selection['levels'], time=selection['time_slice'],
    #                                         latitude=selection['lat_slice'], longitude=selection['long_slice'])
    data = data[selection['variables']].sel(level=selection['levels'],
                                        latitude=selection['lat_slice'], longitude=selection['long_slice'])

    time_size = data.sizes['time']
    level_size = data.sizes['level']
    lat_size = data.sizes['latitude']
    lon_size = data.sizes['longitude']
    feature_size = len(selection['variables'])

    time_arrays = []
    for i, var_name in enumerate(data.data_vars):
        var_data = data[var_name].values
        num_dims = len(var_data.shape)
        if num_dims != 4:
            has_level_size = False
        else: has_level_size = True

        dataset_shape = (time_size, level_size, lat_size, lon_size)
        time_array = np.empty(dataset_shape, dtype=var_data.dtype)
        
        if has_level_size:
            time_array[:] = var_data
        else:
            time_array[:] = np.expand_dims(var_data, axis=1)
        
        time_arrays.append(time_array)

    dataset = np.stack(time_arrays, axis=-1)
    print("processed dataset shape:", dataset.shape)



    #processing to time series
    default_intervals = [-120, -56, -28, -12, -8, -4, -3, -2, -1, 0, 4]
    #todo: handle longer window size

    inputs = []
    labels = []
    for i in range(len(dataset)):
        sequence = []
        for interval in default_intervals:
            index = i+interval
            if 0 <= index < len(dataset):
                sequence.append(dataset[index])
        
        if len(sequence) == len(default_intervals):
            inputs.append(sequence[0:len(sequence)-1])
            labels.append(sequence[-1])
    inputs = np.stack(inputs, axis=0)
    labels = np.stack(labels, axis=0)


    #split into training and testing
    num_samples = len(inputs)
    training_size = math.floor(split_percentage*num_samples)
    X_train = inputs[0:training_size]
    X_test = inputs[training_size::]
    Y_train = labels[0:training_size]
    Y_test = labels[training_size::]


    print("x train:", X_train.shape)
    print("x test:", X_test.shape)
    print("y train:", Y_train.shape)
    print("y test:", Y_test.shape)
    
    return X_train, X_test, Y_train, Y_test



    # np.save('../data/test_data_array.npy', dataset)


preprocess_data(0.8)




