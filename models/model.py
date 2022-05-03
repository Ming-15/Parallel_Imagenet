import torch
import torch.nn as nn

class vgg19_local(nn.Module):
    def __init__(self):
        super(vgg19_local,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            ,nn.ReLU()
            ,nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            , nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            )
        self.classfier = nn.Sequential(
                nn.Linear(in_features=25088, out_features=4096, bias=True)
                ,nn.ReLU()
                ,nn.Dropout(p=0.5)
                ,nn.Linear(in_features=4096, out_features=4096, bias=True)
                ,nn.ReLU()
                ,nn.Dropout(p=0.5)
                ,nn.Linear(in_features=4096, out_features=1000, bias=True)
              )
    def forward(self,x):
        x = self.features(x)
        x = self.classfier(x)
        return x
class vgg19_momdify(nn.Module):
    def __init__(self):
        super(vgg19_momdify,self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(5,5), stride=(1, 1), padding=(2,2))
            ,nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,nn.Conv2d(64, 128, kernel_size=(5,5), stride=(1, 1), padding=(2,2))
            ,nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,nn.Conv2d(128, 256, kernel_size=(5,5), stride=(1, 1), padding=(2,2))
            ,nn.ReLU()
            # ,nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # ,nn.ReLU()
            ,nn.Conv2d(256, 256, kernel_size=(5,5), stride=(1, 1), padding=(2,2))
            ,nn.ReLU()
            # ,nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # ,nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,nn.Conv2d(256, 512, kernel_size=(5,5), stride=(1, 1), padding=(2,2))
            ,nn.ReLU()
            # ,nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # ,nn.ReLU()
            ,nn.Conv2d(512, 512, kernel_size=(5,5), stride=(1, 1), padding=(2,2))
            ,nn.ReLU()
            # ,nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # ,nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            ,nn.Conv2d(512, 512, kernel_size=(5,5), stride=(1, 1), padding=(2,2))
            ,nn.ReLU()
            # ,nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # ,nn.ReLU()
            ,nn.Conv2d(512, 512, kernel_size=(5,5), stride=(1, 1), padding=(2,2))
            ,nn.ReLU()
            # ,nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            # , nn.ReLU()
            ,nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            )
        self.classfier = nn.Sequential(
                nn.Linear(in_features=25088, out_features=4096, bias=True)
                ,nn.ReLU()
                ,nn.Dropout(p=0.5)
                ,nn.Linear(in_features=4096, out_features=4096, bias=True)
                ,nn.ReLU()
                ,nn.Dropout(p=0.5)
                ,nn.Linear(in_features=4096, out_features=1000, bias=True)
              )
    def forward(self,x):
        x = self.features(x)
        x = self.classfier(x)
        return x
class HCNN(nn.Module):
    def __init__(self,output_dim):
        super(HCNN, self).__init__()
        # if we want the unit output
        self.unit_out_tag = False
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 48, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 96, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(384, 768, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.Conv2d(768, 768, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.Conv2d(768, 768, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(768),
            nn.ReLU(),
        )

        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.avgpool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1),))

        self.classifier = nn.Sequential(nn.Linear(768, output_dim), nn.ReLU(), nn.Softmax(dim=1))
        # self.out_layer = nn.Sequential(nn.Linear(1000,5),nn.Softmax())
    def forward(self, x):
        if self.unit_out_tag:
            pass
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.shape[0], 768)
        out = self.classifier(out)
        return out

