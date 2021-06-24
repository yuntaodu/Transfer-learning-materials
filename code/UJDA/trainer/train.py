import tqdm
import argparse
import sys
sys.path.append("..")
from utils.config import Config
from torch.autograd import Variable
import torch
from model.UJDA import UJDA
from model.Logger import Logger
from tensorboardX import SummaryWriter
import numpy



class INVScheduler(object):
    def __init__(self, gamma, decay_rate, init_lr=0.001):
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.init_lr = init_lr

    def next_optimizer(self, group_ratios, optimizer, num_iter):
        lr = self.init_lr * (1 + self.gamma * num_iter) ** (-self.decay_rate)
        i = 0
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * group_ratios
            i += 1
        return optimizer


# ==============eval
def evaluate(model_instance, input_loader):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        probabilities, pro1, pro2 = model_instance.predict(inputs)

        probabilities = probabilities.data.float()
        labels = labels.data.float()
        if first_test:
            all_probs = probabilities
            all_labels = labels
            first_test = False
        else:
            all_probs = torch.cat((all_probs, probabilities), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    _, predict = torch.max(all_probs, 1)
    accuracy = float(torch.sum(torch.squeeze(predict).float() == all_labels)) / float(all_labels.size()[0])

    model_instance.set_train(ori_train_state)
    return {'accuracy': accuracy}


def save_features(model_instance, input_loader, filename):
    ori_train_state = model_instance.is_train
    model_instance.set_train(False)
    num_iter = len(input_loader)
    iter_test = iter(input_loader)
    first_test = True

    for i in range(num_iter):
        data = iter_test.next()
        inputs = data[0]
        labels = data[1]
        if model_instance.use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)
        features = model_instance.get_features(inputs)

        features = features.data.float()
        labels = labels.data.float()

        if first_test:
            all_features = features
            all_labels = labels
            first_test = False
        else:
            all_features = torch.cat((all_features, features), 0)
            all_labels = torch.cat((all_labels, labels), 0)

    all_features = all_features.cpu().numpy()
    all_labels = all_labels.cpu().numpy()

    numpy.savetxt(filename, all_features, fmt='%f', delimiter=' ')
    numpy.savetxt(filename+'_label', all_labels, fmt='%d', delimiter=' ')

    model_instance.set_train(ori_train_state)



def train(model_instance, train_source_loader, train_target_loader, test_target_loader, test_source_loader, group_ratios,
          max_iter, optimizer, eval_interval, lr_scheduler, num_k = 4, iter_classifier = 10000):
    model_instance.set_train(True)
    print("start train...")
    writer = SummaryWriter()
    iter_num = 0
    epoch = 0
    total_progress_bar = tqdm.tqdm(desc='Train iter', total=max_iter)
    optimizer_c_net = optimizer[0]
    optimizer_classifier = optimizer[1]
    optimizer_classifier1 = optimizer[2]
    optimizer_classifier2 = optimizer[3]
    best_acc = 0.0
    while True:
        for (datas, datat) in tqdm.tqdm(
                zip(train_source_loader, train_target_loader),
                total=min(len(train_source_loader), len(train_target_loader)),
                desc='Train epoch = {}'.format(epoch), ncols=80, leave=False):
            inputs_source, labels_source = datas
            inputs_target, labels_target = datat

            optimizer_c_net = lr_scheduler.next_optimizer(group_ratios[0], optimizer_c_net, iter_num / 5)
            optimizer_classifier = lr_scheduler.next_optimizer(group_ratios[1], optimizer_classifier, iter_num / 5)
            optimizer_classifier1 = lr_scheduler.next_optimizer(group_ratios[2], optimizer_classifier1, iter_num / 5)
            optimizer_classifier2 = lr_scheduler.next_optimizer(group_ratios[3], optimizer_classifier2, iter_num / 5)

            optimizer_c_net.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer_classifier1.zero_grad()
            optimizer_classifier2.zero_grad()

            if model_instance.use_gpu:
                inputs_source, inputs_target, labels_source, labels_target = Variable(inputs_source).cuda(), Variable(
                    inputs_target).cuda(), Variable(labels_source).cuda(), Variable(labels_target).cuda()
            else:
                inputs_source, inputs_target, labels_source, labels_target = Variable(inputs_source), Variable(
                    inputs_target), Variable(labels_source), Variable(labels_target)

            #step1:
            if iter_num < iter_classifier:
                iter_num = iter_num + 1

                classifier_loss = model_instance.get_loss_classifier(inputs_source, labels_source)
                joint1_loss, joint2_loss = model_instance.get_loss_source_joint(inputs_source, labels_source)
                loss = classifier_loss + joint1_loss + joint2_loss
                loss.backward()
                optimizer_c_net.step()
                optimizer_classifier.step()
                optimizer_classifier1.step()
                optimizer_classifier2.step()

                optimizer_c_net.zero_grad()
                optimizer_classifier.zero_grad()
                optimizer_classifier1.zero_grad()
                optimizer_classifier2.zero_grad()

                if iter_num % 200 == 0:
                    eval_result_s = evaluate(model_instance, test_source_loader)
                    eval_result_t = evaluate(model_instance, test_target_loader)

                    #save_features(model_instance, test_source_loader, 'source_only')
                    #save_features(model_instance, test_target_loader, 'target_only')

                    print('\n classifier_loss: {:.4f}, joint1_loss:{:.4f}, joint2_loss:{:.4f}, val acc_s: {:.4f}, val acc_t: {:.4f}'.format(
                            classifier_loss, joint1_loss, joint2_loss, eval_result_s['accuracy'], eval_result_t['accuracy']))

                continue

            #step2:
            lambda_t = 0.1
            lambda_svat = 1.0
            lambda_tvat = 10.0

            source_vat = model_instance.vat(inputs_source, 0.5)
            optimizer_c_net.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer_classifier1.zero_grad()
            optimizer_classifier2.zero_grad()

            target_vat = model_instance.vat(inputs_target, 0.5)
            optimizer_c_net.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer_classifier1.zero_grad()
            optimizer_classifier2.zero_grad()

            source_vat_loss = model_instance.get_loss_vat(inputs_source, source_vat)
            target_vat_loss = model_instance.get_loss_vat(inputs_target, target_vat)

            classifier_loss = model_instance.get_loss_classifier(inputs_source, labels_source)
            entropy_loss = model_instance.get_loss_entropy(inputs_target)
            total_loss = classifier_loss +  lambda_svat * source_vat_loss + lambda_t *(entropy_loss + lambda_tvat * target_vat_loss)
            total_loss.backward()
            optimizer_c_net.step()
            optimizer_classifier.step()

            optimizer_c_net.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer_classifier1.zero_grad()
            optimizer_classifier2.zero_grad()


            if iter_num % 200 == 0:
                eval_result_s = evaluate(model_instance, test_source_loader)
                eval_result_t = evaluate(model_instance, test_target_loader)
                print('\n classifier_loss: {:.4f}, entropy_loss:{:.4f}, source_vat_loss:{:.4f}, target_vat_loss:{:.4f}, val acc_s: {:.4f}, val acc_t: {:.4f}'.format(classifier_loss, entropy_loss, source_vat_loss, target_vat_loss, eval_result_s['accuracy'], eval_result_t['accuracy']))


            lambda_dsc = 1.0
            lambda_dtc = 1.0
            lambda_d = 1.0

            joints1_loss, joints2_loss = model_instance.get_loss_source_joint(inputs_source, labels_source)
            jointt1_loss, jointt2_loss = model_instance.get_loss_target_joint(inputs_target)
            loss_dis_s, loss_dis_t = model_instance.get_loss_discrepancy(inputs_source, inputs_target)
            total_loss = lambda_dsc * (joints1_loss + joints2_loss) + lambda_dtc * (jointt1_loss + jointt2_loss) - lambda_d * (loss_dis_s + loss_dis_t)
            total_loss.backward()
            optimizer_classifier1.step()
            optimizer_classifier2.step()
            optimizer_c_net.zero_grad()
            optimizer_classifier.zero_grad()
            optimizer_classifier1.zero_grad()
            optimizer_classifier2.zero_grad()

            if iter_num % 200 == 0:
                print(
                    '\n joints1_loss: {:.4f}, joints2_loss:{:.4f}, jointt1_loss:{:.4f}, jointt2_loss:{:.4f}, loss_dis_s: {:.4f}, loss_dis_t: {:.4f}'.format(
                        joints1_loss, joints2_loss, jointt1_loss, jointt2_loss, loss_dis_s,loss_dis_t))

            #step3:
            lambda_dsa = 0.1
            lambda_dta = 0.1

            for i in range(num_k):
                joints1_loss, joints2_loss, jointt1_loss, jointt2_loss = model_instance.get_loss_adv(inputs_source, inputs_target, labels_source)
                loss_dis_s, loss_dis_t = model_instance.get_loss_discrepancy(inputs_source, inputs_target)
                total_loss = lambda_dsa*(joints1_loss + joints2_loss) + lambda_dta*(jointt1_loss + jointt2_loss) + loss_dis_s + loss_dis_t
                total_loss.backward()
                optimizer_c_net.step()
                optimizer_c_net.zero_grad()
                optimizer_classifier.zero_grad()
                optimizer_classifier1.zero_grad()
                optimizer_classifier2.zero_grad()

            # val
            if iter_num % eval_interval == 0 and iter_num != 0:
                eval_result = evaluate(model_instance, test_target_loader)
                if eval_result['accuracy'] > best_acc:
                    best_acc = eval_result['accuracy']
                    #save_features(model_instance, test_source_loader, 'source')
                    #save_features(model_instance, test_target_loader, 'target')
                print(
                    '\n joints1_loss: {:.4f}, joints2_loss:{:.4f}, jointt1_loss:{:.4f}, jointt2_loss:{:.4f}, loss_dis_s: {:.4f}, loss_dis_t: {:.4f}, current accuracy : {:.4f}, best accuracy:{}'.format(
                        joints1_loss, joints2_loss, jointt1_loss, jointt2_loss, loss_dis_s, loss_dis_t, eval_result['accuracy'], best_acc))

            iter_num += 1
            total_progress_bar.update(1)
        epoch += 1
        if iter_num >= max_iter:
            break
    print('finish train')
    writer.close()


if __name__ == '__main__':

    from preprocess.data_provider import load_images
    import os

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='all sets of configuration parameters',
                        default='../config/dann.yml')
    parser.add_argument('--dataset', default='Office-31', type=str,
                        help='which dataset')
    parser.add_argument('--src_address', default=None, type=str,
                        help='address of image list of source dataset')
    parser.add_argument('--tgt_address', default=None, type=str,
                        help='address of image list of target dataset')
    parser.add_argument('--src_test_address', default=None, type=str,
                        help='address of image list of source test dataset')

    args = parser.parse_args()

    cfg = Config(args.config)

    source_file = args.src_address
    target_file = args.tgt_address
    source_file_test = args.src_test_address

    if args.dataset == 'Office-31':
        class_num = 31
        width = 1024
        srcweight = 4
        is_cen = False
    elif args.dataset == 'image-clef':
        class_num = 12
        width = 1024
        srcweight = 4
        is_cen = False
    else:
        width = -1


    model_instance = UJDA(use_base= True, base_net='ResNet50', use_gpu= True, class_num=class_num)
    train_source_loader = load_images(source_file, batch_size= cfg.batch_size, is_cen=is_cen)
    train_target_loader = load_images(target_file, batch_size= cfg.batch_size, is_cen=is_cen)
    test_source_loader = load_images(source_file_test, batch_size= cfg.batch_size, is_cen=is_cen)
    test_target_loader = load_images(target_file, batch_size= cfg.batch_size, is_train=False)

    param_groups = model_instance.get_parameter_list()
    group_ratios = [group['lr'] for group in param_groups]

    optimizer_c_net = torch.optim.SGD(model_instance.c_net.base_network.parameters(), lr = cfg.init_lr_c_net, weight_decay = 0.0005, momentum = 0.9)
    optimizer_classifier = torch.optim.SGD(model_instance.c_net.classifier.parameters(), lr= cfg.init_lr_classifier,weight_decay = 0.0005, momentum = 0.9)
    optimizer_classifier1 = torch.optim.SGD(model_instance.c_net.classifier1.parameters(), lr= cfg.init_lr_classifier, weight_decay = 0.0005, momentum = 0.9)
    optimizer_classifier2 = torch.optim.SGD(model_instance.c_net.classifier2.parameters(), lr= cfg.init_lr_classifier, weight_decay = 0.0005, momentum = 0.9)

    optimizer = [optimizer_c_net, optimizer_classifier, optimizer_classifier1, optimizer_classifier2]
    lr_scheduler = INVScheduler(gamma=cfg.lr_scheduler.gamma,
                                decay_rate=cfg.lr_scheduler.decay_rate,
                                init_lr=cfg.init_lr)

    train(model_instance, train_source_loader, train_target_loader, test_target_loader, test_source_loader, group_ratios,
              max_iter=100000, optimizer=optimizer, eval_interval=200, lr_scheduler = lr_scheduler)

