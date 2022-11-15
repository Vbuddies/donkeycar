class KerasSensors(KerasPilot):
    def __init__(self, input_shape=(120, 160, 3), num_sensors=2):
        super().__init__()
        self.num_sensors = num_sensors
        self.model = self.create_model(input_shape)

    def create_model(self, input_shape):
        drop = 0.2
        img_in = Input(shape=input_shape, name='img_in')
        x = core_cnn_layers(img_in, drop)
        x = Dense(100, activation='relu', name='dense_1')(x)
        x = Dropout(drop)(x)
        x = Dense(50, activation='relu', name='dense_2')(x)
        x = Dropout(drop)(x)
        # up to here, this is the standard linear model, now we add the
        # sensor data to it
        sensor_in = Input(shape=(self.num_sensors, ), name='sensor_in')
        y = sensor_in
        z = concatenate([x, y])
        # here we add two more dense layers
        z = Dense(50, activation='relu', name='dense_3')(z)
        z = Dropout(drop)(z)
        z = Dense(50, activation='relu', name='dense_4')(z)
        z = Dropout(drop)(z)
        # two outputs for angle and throttle
        outputs = [
            Dense(1, activation='linear', name='n_outputs' + str(i))(z)
            for i in range(2)]

        # the model needs to specify the additional input here
        model = Model(inputs=[img_in, sensor_in], outputs=outputs)
        return model

    def compile(self):
        self.model.compile(optimizer=self.optimizer, loss='mse')

    def inference(self, img_arr, other_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        sens_arr = other_arr.reshape((1,) + other_arr.shape)
        outputs = self.model.predict([img_arr, sens_arr])
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]

    def x_transform(self, record: TubRecord) -> XY:
        img_arr = super().x_transform(record)
        # for simplicity we assume the sensor data here is normalised
        sensor_arr = np.array(record.underlying['sensor'])
        # we need to return the image data first
        return img_arr, sensor_arr

    def x_translate(self, x: XY) -> Dict[str, Union[float, np.ndarray]]:
        assert isinstance(x, tuple), 'Requires tuple as input'
        # the keys are the names of the input layers of the model
        return {'img_in': x[0], 'sensor_in': x[1]}

    def y_transform(self, record: TubRecord):
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return angle, throttle

    def y_translate(self, y: XY) -> Dict[str, Union[float, np.ndarray]]:
        if isinstance(y, tuple):
            angle, throttle = y
            # the keys are the names of the output layers of the model
            return {'n_outputs0': angle, 'n_outputs1': throttle}
        else:
            raise TypeError('Expected tuple')

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shape()[1:]
        # the keys need to match the models input/output layers
        shapes = ({'img_in': tf.TensorShape(img_shape),
                   'sensor_in': tf.TensorShape([self.num_sensors])},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes