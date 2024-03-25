import pickle


class Preprocessing:
    def __init__(self, data_analytics=None, preprocessing=None):
        if isinstance(data_analytics, list):
            self.data_analytics = data_analytics
        else:
            if data_analytics is not None:
                self.data_analytics = [data_analytics]
            else:
                self.data_analytics = []
        if isinstance(preprocessing, list):
            self.preprocessing = preprocessing
        else:
            if preprocessing is not None:
                self.preprocessing = [preprocessing]
            else:
                self.preprocessing = []

    def fit(self, X, y=None):
        for method in self.preprocessing:
            if y is not None:
                method.fit(X, y)
            else:
                method.fit(X)
        self.fitted = True

    def fit_transform(self, X, y=None):
        for method in self.preprocessing:
            if y is not None:
                method.fit_transform(X, y)
            else:
                method.fit_transform(X)
        self.fitted = True
        if y is not None:
            return X, y
        return X

    def transform(self, X, y=None):
        if self.fitted:
            for method in self.preprocessing:
                if y is not None:
                    method.transform(X, y)
                else:
                    method.transform(X)
        else:
            raise ValueError("Methods to transform the data are not trained")
        if y is not None:
            return X, y
        return X

    def apply_analytics(self, X, y=None):
        # TODO : Future work
        pass

    def add_method_data_analytics(self, data_analytics):
        self.data_analytics.append(data_analytics)

    def add_method_preprocessing(self, preprocessing):
        self.preprocessing.append(preprocessing)

    def save_model(self, key=None, path=None):
        if key is not None:
            pass
        else:
            if self.fitted:
                for method in self.preprocessing:
                    if path is not None:
                        with open(path+str(method)+'pickle', 'wb') as file:
                            pickle.dump(method, file)
                    else:
                        with open('./'+str(method)+'pickle', 'wb') as file:
                            pickle.dump(method, file)
            else:
                raise ValueError("To save the methods they should be trained")

    def load_model(self, path):
        pass

    def read(self, path):
        pass

    def write(self, path):
        pass
