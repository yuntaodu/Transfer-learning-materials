import torch.nn as nn
import model.backbone as backbone
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable

class GradientReverseLayer(torch.autograd.Function):
    def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter

    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    def backward(self, grad_output):
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) -
            (self.high_value - self.low_value) + self.low_value)
        return -self.coeff * grad_output

class UJDAnet(nn.Module):

    def __init__(self, use_base = False, base_net = 'ResNet50', class_num = 31 ,bottleneck_dim = 1024, width =1024):
        super(UJDAnet, self).__init__()
        self.use_base = use_base
        self.bottleneck_dim = bottleneck_dim
        self.class_num = class_num
        if use_base:
            self.base_network = backbone.network_dict[base_net]()
        else:
            self.base_network_conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
            self.base_network_bn1 = nn.BatchNorm2d(32)
            self.base_network_conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=2, padding=2)
            self.base_network_bn2 = nn.BatchNorm2d(48)
            self.base_network_conv3 = nn.Conv2d(48, 64, kernel_size=5, stride=2, padding=2)
            self.base_network_bn3 = nn.BatchNorm2d(64)
            self.base_network_fc1 = nn.Linear(3136, bottleneck_dim)
            self.base_network_bn1_fc = nn.BatchNorm1d(bottleneck_dim)

        self.classifier_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.ReLU(), nn.Dropout(0.5),
                                       nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                       nn.Linear(width, class_num)]
        self.classifier = nn.Sequential(*self.classifier_layer_list)
        self.classifier1_layer_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.ReLU(), nn.Dropout(0.5),
                                       nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                       nn.Linear(width, 2*class_num)]
        self.classifier1 = nn.Sequential(*self.classifier1_layer_list)
        self.classifier_layer_2_list = [nn.Linear(self.base_network.output_num(), bottleneck_dim), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, 2*class_num)]
        self.classifier2 = nn.Sequential(*self.classifier_layer_2_list)

        ## init

        for dep in range(3):
            self.classifier[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier[dep * 3].bias.data.fill_(0.0)
            self.classifier1[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier1[dep * 3].bias.data.fill_(0.0)
            self.classifier2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier2[dep * 3].bias.data.fill_(0.0)


        self.softmax = nn.Softmax(dim = 1)
        self.grl_layer = GradientReverseLayer()

        self.parameter_list = [{'params':self.base_network.parameters(), 'lr':0.1},
                               {'params':self.classifier.parameters(), 'lr':1},
                               {'params':self.classifier1.parameters(), 'lr':1},
                               {'params':self.classifier2.parameters(),'lr':1}]

    def forward(self, x):
        if self.use_base:
            features = self.base_network(x)
        else:
            x = F.max_pool2d(F.relu(self.base_network_bn1(self.base_network_conv1(x))), stride=2, kernel_size=3, padding=1)
            x = F.max_pool2d(F.relu(self.base_network_bn2(self.base_network_conv2(x))), stride=2, kernel_size=3, padding=1)
            x = F.relu(self.base_network_bn3(self.base_network_conv3(x)))
            x = x.view(x.size(0), 3136)
            x = F.relu(self.base_network_bn1_fc(self.base_network_fc1(x)))
            features = F.dropout(x, training=self.training)

        outputs_classifier = self.classifier(features)
        outputs_classifier1 = self.classifier1(features)
        outputs_classifier2 = self.classifier2(features)

        return features, outputs_classifier, outputs_classifier1, outputs_classifier2


class UJDA(object):

    def __init__(self, use_base = False, base_net = 'ResNet50', class_num = 31, use_gpu = True):
        self.c_net = UJDAnet(use_base= use_base, base_net = base_net, class_num = class_num)
        self.use_gpu = use_gpu
        self.is_train =False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1, dim =1) - F.softmax(out2, dim = 1)))

    def vat(self, inputs, radius):
        eps = Variable(torch.randn(inputs.data.size()).cuda())
        eps_norm = 1e-6 *(eps/torch.norm(eps,dim=(2,3),keepdim=True))
        eps = Variable(eps_norm.cuda(),requires_grad=True)
        _, outputs_classifier1, _, _ = self.c_net(inputs)
        _, outputs_classifier2, _, _ = self.c_net(inputs + eps)
        loss_p = self.discrepancy(outputs_classifier1,outputs_classifier2)
        loss_p.backward(retain_graph=True)

        eps_adv = eps.grad
        eps_adv = eps_adv/torch.norm(eps_adv)
        image_adv = inputs + radius * eps_adv

        return image_adv

    def get_loss_vat(self, inputs, inputs_vat):
        _, outputs_classifier1, _, _ = self.c_net(inputs)
        _, outputs_classifier2, _, _ = self.c_net(inputs_vat)

        vat_loss = self.discrepancy(outputs_classifier1, outputs_classifier2)

        return vat_loss

    def get_loss_entropy(self, inputs):
        features, outputs_classifier, outputs_classifier1, outputs_classifier2 = self.c_net(inputs)

        output = F.softmax(outputs_classifier, dim=1)
        entropy_loss = - torch.mean(output * torch.log(output + 1e-6))

        return entropy_loss

    def get_loss_classifier(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss().cuda()
        features, outputs_classifier, outputs_classifier1, outputs_classifier2 = self.c_net(inputs)

        classifier_loss = class_criterion(outputs_classifier, labels_source)

        return classifier_loss

    def get_loss_source_joint(self, inputs_source, labels_source):
        class_criterion = nn.CrossEntropyLoss().cuda()
        _, _, outputs_classifier_s1, outputs_classifier_s2 = self.c_net(inputs_source)

        classifier1_loss_s = class_criterion(outputs_classifier_s1, labels_source)
        classifier2_loss_s = class_criterion(outputs_classifier_s2, labels_source)

        return classifier1_loss_s, classifier2_loss_s


    def get_loss_target_joint(self, inputs_target):
        class_criterion = nn.CrossEntropyLoss().cuda()

        _, outputs_classifier_t, outputs_classifier_t1, outputs_classifier_t2 = self.c_net(inputs_target)

        labels_target = outputs_classifier_t.data.max(1)[1] + self.class_num

        classifier1_loss_t = class_criterion(outputs_classifier_t1, labels_target)
        classifier2_loss_t = class_criterion(outputs_classifier_t2, labels_target)

        return classifier1_loss_t, classifier2_loss_t

    def get_loss_adv(self, inputs_source, inputs_target, labels_source):
        class_criterion = nn.CrossEntropyLoss().cuda()
        _, _, outputs_classifier_s1, outputs_classifier_s2 = self.c_net(inputs_source)

        labels_source0 = labels_source + self.class_num
        classifier1_loss_s = class_criterion(outputs_classifier_s1, labels_source0)
        classifier2_loss_s = class_criterion(outputs_classifier_s2, labels_source0)

        _, outputs_classifier_t, outputs_classifier_t1, outputs_classifier_t2 = self.c_net(inputs_target)

        labels_target = outputs_classifier_t.data.max(1)[1]
        classifier1_loss_t = class_criterion(outputs_classifier_t1, labels_target)
        classifier2_loss_t = class_criterion(outputs_classifier_t2, labels_target)

        return  classifier1_loss_s, classifier2_loss_s, classifier1_loss_t, classifier2_loss_t

    def get_loss_discrepancy(self, inputs_source, inputs_target):
        _, _, outputs_classifier_s1, outputs_classifier_s2 = self.c_net(inputs_source)

        _, _, outputs_classifier_t1, outputs_classifier_t2 = self.c_net(inputs_target)

        loss_dis_s = self.discrepancy(outputs_classifier_s1, outputs_classifier_s2)
        loss_dis_t = self.discrepancy(outputs_classifier_t1, outputs_classifier_t2)

        return loss_dis_s, loss_dis_t

    def predict(self, inputs):
        _, outputs_classifier_t, outputs_classifier1, outputs_classifier2 = self.c_net(inputs)
        return F.softmax(outputs_classifier_t, dim = 1), F.softmax(outputs_classifier1, dim = 1), F.softmax(outputs_classifier2, dim = 1)

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

    def get_features(self,inputs):
        features, outputs_classifier, outputs_classifier1, outputs_classifier2 = self.c_net(inputs)
        return features

    def get_parameter_list(self):
        return self.c_net.parameter_list
