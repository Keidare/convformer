
import utils.train 
import utils.evaluate

def train_unet():
    logs_root = 'logs/cityscapes/022025UNETResnet34'
    data_root='data/cityscapes'
    learning_rate=0.00006
    weight_decay=0.0001
    batch_size=4
    size = (1024,512)
    epochs = 35
    num_workers=4
    backbone='resnet34'
    model_name='unet-new'
    num_classes=20
    utils.train.train(logs_root, data_root, learning_rate, size, weight_decay, batch_size, epochs, num_workers, backbone, model_name, num_classes)

def evaluate_unet():
    data_root='./data/cityscapes'
    model_path='./logs/cityscapes/022025UNETEfficientNet/models/model_029.pt'
    size=(1024, 512)
    batch_size=4
    num_workers=4
    num_classes=20
    backbone='efficientnet-b0'
    model_name='unet-new'
    utils.evaluate.evaluate_cityscapes(data_root, model_path, size, batch_size, num_workers, num_classes, backbone, model_name)

def train_segformer():
    logs_root = 'logs/cityscapes/022025Segformer'
    data_root='data/cityscapes'
    learning_rate=0.00006
    weight_decay=0.0001
    batch_size=4
    epochs=35
    size=(1024, 512)
    num_workers=4
    backbone='mit_b1'
    model_name='vit'
    num_classes=20
    utils.train.train(logs_root, data_root, learning_rate, size, weight_decay, batch_size, epochs, num_workers, backbone, model_name, num_classes)

def evaluate_segformer():
    data_root='./data/cityscapes'
    model_path='./logs/cityscapes/022025Segformer/models/model_021.pt'
    size=(1024, 512)
    batch_size=4
    num_workers=4
    num_classes=20
    backbone='mit_b1'
    model_name='vit'
    utils.evaluate.evaluate_cityscapes(data_root, model_path, size, batch_size, num_workers, num_classes, backbone, model_name)    



if __name__ == '__main__':
    # train_segformer()
    evaluate_unet()
    # print('-'*50)
    # evaluate_segformer()

#  best model effnet = 029, resnet = 032, segformer = 021