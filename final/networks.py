import torch

class FlowRegClf_2D(torch.nn.Module):
    def __init__(self,
                 inchannels,
                 sample_input,
                 device):
        super(FlowRegClf_2D, self).__init__()

        self.sample_input = sample_input
        self.device = device

        # Define activation functions
        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.nn.functional.sigmoid

        # Define convolutional kernels
        self.conv1 = torch.nn.Conv2d(inchannels, 8, 16, device=self.device)
        self.conv2 = torch.nn.Conv2d(8, 16, 8, device=self.device)
        self.conv3 = torch.nn.Conv2d(16, 32, 5, device=self.device)
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.flattenlen = self._getflatlen()

        # Define classification layers
        '''
        self.classify1 = torch.nn.Linear(self.flattenlen, 128, device=self.device)
        self.classify2 = torch.nn.Linear(128, 128, device=self.device)
        self.classify3 = torch.nn.Linear(128, 4, device=self.device)
        '''
        self.classify1 = torch.nn.Linear(self.flattenlen, 64, device=self.device)
        self.classify2 = torch.nn.Linear(64, 64, device=self.device)
        self.classify3 = torch.nn.Linear(64, 4, device=self.device)

        # Define velocity regression layers
        self.vel1 = torch.nn.Linear(self.flattenlen, 64, device=self.device)
        self.vel2 = torch.nn.Linear(64, 64, device=self.device)
        self.vel3 = torch.nn.Linear(64, 1, device=self.device)

        # Define length regression layers
        self.len1 = torch.nn.Linear(self.flattenlen, 64, device=self.device)
        self.len2 = torch.nn.Linear(64, 64, device=self.device)
        self.len3 = torch.nn.Linear(64, 1, device=self.device)

    def _getflatlen(self):
        with torch.no_grad():
            x = self._conv(self.sample_input)
            l = 1
            for i in range(len(x.shape)):
                l *= x.shape[i]
        return l

    def _conv(self, tens):
        x = self.pool(self.relu(self.conv1(tens)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x

    def _classify(self, tens):
        x = self.relu(self.classify1(tens))
        x = self.relu(self.classify2(x))
        x = self.classify3(x)
        return x

    def _velreg(self, tens):
        x = self.relu(self.vel1(tens))
        x = self.relu(self.vel2(x))
        x = self.vel3(x)
        return x

    def _lenreg(self, tens):
        x = self.relu(self.len1(tens))
        x = self.relu(self.len2(x))
        x = self.len3(x)
        return x

    def forward(self, tens):
        flat = self._conv(tens)
        flat = flat.view(-1, self.flattenlen)

        class_pred = self._classify(flat)
        vel_reg = self._velreg(flat)
        len_reg = self._lenreg(flat)
        return class_pred, vel_reg, len_reg

class FlowRegClf_1D(torch.nn.Module):
    def __init__(self,
                 inchannels,
                 sample_input,
                 device):
        super(FlowRegClf_1D, self).__init__()

        self.sample_input = sample_input
        self.device = device

        # Define activation functions
        self.relu = torch.nn.functional.relu
        self.sigmoid = torch.nn.functional.sigmoid

        # Define convolutional kernels
        self.conv1 = torch.nn.Conv1d(inchannels, 8, 16, device=self.device)
        self.conv2 = torch.nn.Conv1d(8, 16, 8, device=self.device)
        self.conv3 = torch.nn.Conv1d(16, 32, 5, device=self.device)
        self.pool = torch.nn.MaxPool1d(2, 2)

        self.flattenlen = self._getflatlen()

        # Define classification layers
        '''
        self.classify1 = torch.nn.Linear(self.flattenlen, 128, device=self.device)
        self.classify2 = torch.nn.Linear(128, 128, device=self.device)
        self.classify3 = torch.nn.Linear(128, 4, device=self.device)
        '''
        self.classify1 = torch.nn.Linear(self.flattenlen, 64, device=self.device)
        self.classify2 = torch.nn.Linear(64, 64, device=self.device)
        self.classify3 = torch.nn.Linear(64, 4, device=self.device)

        # Define velocity regression layers
        self.vel1 = torch.nn.Linear(self.flattenlen, 64, device=self.device)
        self.vel2 = torch.nn.Linear(64, 64, device=self.device)
        self.vel3 = torch.nn.Linear(64, 1, device=self.device)

        # Define length regression layers
        self.len1 = torch.nn.Linear(self.flattenlen, 64, device=self.device)
        self.len2 = torch.nn.Linear(64, 64, device=self.device)
        self.len3 = torch.nn.Linear(64, 1, device=self.device)

    def _getflatlen(self):
        with torch.no_grad():
            x = self._conv(self.sample_input)
            l = 1
            for i in range(len(x.shape)):
                l *= x.shape[i]
        return l

    def _conv(self, tens):
        x = self.pool(self.relu(self.conv1(tens)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        return x

    def _classify(self, tens):
        x = self.relu(self.classify1(tens))
        x = self.relu(self.classify2(x))
        x = self.classify3(x)
        return x

    def _velreg(self, tens):
        x = self.relu(self.vel1(tens))
        x = self.relu(self.vel2(x))
        x = self.vel3(x)
        return x

    def _lenreg(self, tens):
        x = self.relu(self.len1(tens))
        x = self.relu(self.len2(x))
        x = self.len3(x)
        return x

    def forward(self, tens):
        flat = self._conv(tens)
        flat = flat.view(-1, self.flattenlen)

        class_pred = self._classify(flat)
        vel_reg = self._velreg(flat)
        len_reg = self._lenreg(flat)
        return class_pred, vel_reg, len_reg
