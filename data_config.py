
class DataConfig:
    data_name = ""
    root_dir = ""
    label_transform = ""
    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'LEVIR':
            self.label_transform = "norm"
            self.root_dir = './YGdata/LEVIR256/'
        elif data_name == 'Google':
            self.label_transform = "norm"
            self.root_dir = './YGdata/Google/'
        elif data_name == 'WHU':
            self.label_transform = "norm"
            self.root_dir = './YGdata/WHU/'
        elif data_name == 'LEVIR_hotmap':
            self.label_transform = "norm"
            self.root_dir = './YGdata/LEVIR_hotmap/'
        elif data_name == 'WHU_hotmap':
            self.label_transform = "norm"
            self.root_dir = './YGdata/WHU_hotmap/'
        elif data_name == 'Google_hotmap':
            self.label_transform = "norm"
            self.root_dir = './YGdata/Google_hotmap/'
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


if __name__ == '__main__':
    data = DataConfig().get_data_config(data_name='LEVIR')
    print(data.data_name)
    print(data.root_dir)
    print(data.label_transform)

