import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from utils import my_utils

NUM_CLASSES = 1


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instances
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=512,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
        # My Additions
        chunk_low_num=config['data_loader']['args']['chunk_low_num'],
        chunk_high_num=config['data_loader']['args']['chunk_high_num'],
        partial_change=config['data_loader']['args']['partial_change'],
        layer_change_lim=config['data_loader']['args']['layer_change_lim']

    )

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config['loss'])
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    # checkpoint = torch.load(config.resume)    #ORIG - dict with state dict in it - not my format
    # state_dict = checkpoint['state_dict']
    state_dict = torch.load(config.resume)
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    from torchsummary import summary

    summary(model, (1, 121, 11, 21))

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    # Accumulate results for histograms
    tot_bias = torch.Tensor().to(device)
    tot_out = torch.Tensor().to(device)
    tot_target = torch.Tensor().to(device)
    tot_sums = torch.Tensor().to(device
                                 )
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            en_dep, x_y, EN = data
            en_dep, x_y, EN = en_dep.to(device), x_y.to(device), EN.to(device)

            target = []
            if NUM_CLASSES == 1:
                target = EN.float().unsqueeze(1)
            elif NUM_CLASSES == 2:
                target = x_y
            elif NUM_CLASSES == 3:
                target = torch.cat([x_y, EN], dim=1)

            output = model(en_dep)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            bias = target - output
            tmp_sum = torch.sum(en_dep, dim=[1, 2, 3, -1])  # Sum the entire energy on the ECAL

            # Accum
            tot_bias = torch.cat((tot_bias, bias), 0)
            tot_target = torch.cat((tot_target, target), 0)
            tot_out = torch.cat((tot_out, output), 0)
            tot_sums = torch.cat((tot_sums, tmp_sum), 0)

            batch_size = en_dep.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    my_utils.plot_and_save_figs(tot_bias, tot_out, tot_target, tot_sums,
                                config['data_loader']['args']['partial_change'],
                                config['data_loader']['args']['layer_change_lim'])

    n_samples = len(data_loader.sampler)
    log = {'loss': total_loss / n_samples}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)

