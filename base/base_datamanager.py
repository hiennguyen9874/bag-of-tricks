class BaseDataManger(object):
    def train_dataloader(self):
        return None
        
    def val_dataloader(self):
        return None
        
    def test_dataloader(self):
        return None
        
    def get_len_train(self):
        return None
        
    def get_len_val(self):
        return None
        
    def get_test_set(self):
        return None
        