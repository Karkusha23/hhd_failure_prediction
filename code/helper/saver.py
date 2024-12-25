import json, os, datetime

class Saver:
    @classmethod
    def save(cls, model, preprocessor, metrics, colab=False, save_csv=True):
        timestamp = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-')[:-7]
        if colab:
            saved_dir = '.\\real-hdd-failure\\code\\helper\\saved'
            curdir = saved_dir + '\\' + str(model) + timestamp
        else:
            saved_dir = os.path.dirname(os.path.realpath(__file__)) + '\\saved'
            curdir = saved_dir + '\\' + str(model) + timestamp
        os.makedirs(curdir)
        data = {'timestamp': timestamp, 
                'model': {'model_name': model.get_model_names(), 'hyperparams': model.get_hyperparams()},
                'preprocess_operations': preprocessor.operations, 
                'metrics': metrics}
        with open(curdir + '\\info.json', 'wt') as outfile:
            outfile.write(json.dumps(data, indent=4).replace('NaN', '"NaN"'))
        if save_csv:
            preprocessor.train_df.to_csv(curdir + '\\dataset.csv')
        model.save_model(curdir + '\\model')
    
