import os
import logging
import torch

from torchsummary import summary

from models import PCB
from data import DataManger
from logger import setup_logging
from evaluators import feature_extractor

def extract_appearance_feature(logger, config, datamanager: DataManger):
    use_gpu = config['n_gpu'] > 0 and torch.cuda.is_available()
    device = torch.device('cuda:0' if use_gpu else 'cpu')

    model = PCB(
        num_classes=datamanager.datasource.get_num_classes('train'),
        num_part=config['model']['num_part'],
        is_training=False)

    logger.info('Loading checkpoint: {} ...'.format(config['resume']))
    checkpoint = torch.load(config['resume'], map_location='cpu')
    
    model.load_state_dict(checkpoint['state_dict'])
    model = model.eval()

    test_loader = datamanager.test_dataloader()
    
    logger.info('Extract feature from query set...')
    query_feature, query_label, query_cam, query_frame = feature_extractor(model, datamanager.test_dataloader()['query'], device)

    logger.info('Extract feature from gallery set...')
    gallery_feature, gallery_label, gallery_cam, gallery_frame = feature_extractor(model, datamanager.test_dataloader()['gallery'], device)

    query_embeddings = (query_feature, query_label, query_cam, query_frame)
    gallery_embeddings = (gallery_feature, gallery_label, gallery_cam, gallery_frame)

    os.makedirs(config['testing']['ouput_dir'], exist_ok=True)

    with open(os.path.join(config['testing']['ouput_dir'], 'query_embeddings.pt'), 'wb') as f:
        torch.save(query_embeddings, f)

    with open(os.path.join(config['testing']['ouput_dir'], 'gallery_embeddings.pt'), 'wb') as f:
        torch.save(gallery_embeddings, f)