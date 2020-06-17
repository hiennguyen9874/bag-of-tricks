import argparse
import os
import logging
import torch

from models import PCB
from data import DataManger
from logger import setup_logging
from utils import read_json, write_json
from testing import extract_appearance_feature, gen_st_distribution, joint_metrics
from evaluators import top_k, mAP, compute_distance_matrix, cmc_rank, feature_extractor, plot_loss, show_image

def main(config):
    setup_logging(os.getcwd())
    logger = logging.getLogger('test')
    
    datamanager = DataManger(config['data'], phase='test')
    
    # extract appearance feature
    # extract_appearance_feature(logger, config, datamanager)

    # generate spatial-temporal distribution
    # gen_st_distribution(logger, config, datamanager)
    
    # joint metric
    joint_metrics(logger, config, datamanager)

    # evaluator without Spatial-temporal Stream
    gallery_feature, gallery_label, gallery_cam, gallery_frames = torch.load(os.path.join(
        config['testing']['ouput_dir'], 'gallery_embeddings.pt'), map_location='cpu')

    query_feature, query_label, query_cam, query_frame = torch.load(os.path.join(
        config['testing']['ouput_dir'], 'query_embeddings.pt'), map_location='cpu')

    # normalize feature vector
    norm = query_feature.norm(p=2, dim=1, keepdim=True)
    query_feature = query_feature.div(norm.expand_as(query_feature))

    norm = gallery_feature.norm(p=2, dim=1, keepdim=True)
    gallery_feature = gallery_feature.div(norm.expand_as(gallery_feature))
    
    distance = compute_distance_matrix(query_feature, gallery_feature)
    
    top1 = top_k(distance, output=gallery_label, target=query_label, k=1)
    top5 = top_k(distance, output=gallery_label, target=query_label, k=5)
    top10 = top_k(distance, output=gallery_label, target=query_label, k=10)
    m_ap = mAP(distance, output=gallery_label, target=query_label, k='all')

    logger.info('without spatial-temporal: top1: {}, top5: {}, top10: {}, mAP: {}'.format(top1, top5, top10, m_ap))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-c', '--config', default='config.json', type=str, help='config file path (default: ./config.json)')
    parser.add_argument('-r', '--resume', default='', type=str, help='resume file path (default: .)')
    parser.add_argument('-e', '--extract', default=True, type=lambda x: (str(x).lower() == 'true'), help='extract feature (default: true')
    args = parser.parse_args()

    config = read_json(args.config)
    config.update({'resume': args.resume})
    config.update({'extract': args.extract})
    
    main(config)