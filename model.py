import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from bayes_opt import BayesianOptimization
import exp
import dataset

class ConvLSTM(nn.Module):
	
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_filters, filter_size, batch_size, dropout, use_bn, window_len):

        super(ConvLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.batch_size = batch_size

        self.dropout = dropout
        self.use_bn = use_bn
        self.window_len = window_len

        # Layers 2-5
        self.conv1 = nn.Conv1d(input_dim, num_filters, filter_size)
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_size)
        self.conv3 = nn.Conv1d(num_filters, num_filters, filter_size)
        self.conv4 = nn.Conv1d(num_filters, num_filters, filter_size)

        # Layers 6-7 - each dense layer has LSTM cells
        self.lstm1 = nn.LSTM(num_filters, hidden_dim, num_layers)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.lstm3 = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.lstm4 = nn.LSTM(hidden_dim, hidden_dim, num_layers)
        self.lstm5 = nn.LSTM(hidden_dim, hidden_dim, num_layers)

        # Layer 9 - prepare for softmax
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        # Layer 1 - flatten (see -1)
        x = x.view(-1, self.input_dim, self.window_len)

        # Layers 2-5 - RELU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Layers 5 and 6 - flatten
        x = x.view(1, -1, self.num_filters)

        # Layers 6-8 - hidden layers
        x, self.hidden = self.lstm1(x, self.hidden)
        x, self.hidden = self.lstm2(x, self.hidden)
        x, self.hidden = self.lstm3(x, self.hidden)
        x, self.hidden = self.lstm4(x, self.hidden)
        x, self.hidden = self.lstm5(x, self.hidden)

        # Layers 8 - flatten, fully connected for softmax. Not sure what dropout does here
        x = x.contiguous().view(-1, self.hidden_dim)
        x = self.dropout(x)
        x = self.fc(x)

        # View flattened output layer
        out = x.view(self.batch_size, -1, self.output_dim)[:, -1, :]

        return out

    def init_hidden(self):
        '''
        Initializes hidden state

        Create two new tensors with sizes n_layers x batch_size x n_hidden,
        initialized to zero, for hidden state and cell state of LSTM
        '''
        weight = next(self.parameters()).data

        # changed this from batch_size to 3*batch_size
        hidden = (weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_(),
        weight.new(self.num_layers, self.batch_size, self.hidden_dim).zero_())

        return hidden

class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, batch_size, dropout, use_bn):

        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.batch_size = batch_size
        self.dropout = dropout
        self.use_bn = use_bn

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)
        self.hidden = self.init_hidden()
        self.fc = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
        torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, x):
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        out = self.fc(lstm_out[:, -1])
        return out

class CNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_filters, filter_size, batch_size, dropout):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.batch_size = batch_size
        self.dropout = dropout

        self.conv1 = nn.Conv1d(input_dim, num_filters, filter_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv1d(num_filters, num_filters, filter_size)
        self.conv3 = nn.Conv1d(num_filters, num_filters, filter_size)

        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim, self.window_len)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(1, -1, self.num_filters)
        x = self.dropout(x)
        x = self.fc(x)
        out = x.view(self.batch_size, -1, self.output_dim)[:, -1, :]

        return out

class Manager():
    def __init__(self, args):
        self.trainset = dataset.NumDataset('data/FakeData.csv', args.x_frames, args.y_frames, args.str_len)
        self.valset = dataset.NumDataset('data/FakeData.csv', args.x_frames, args.y_frames, args.str_len)
        self.testset = dataset.NumDataset('data/FakeData.csv', args.x_frames, args.y_frames, args.str_len)
        self.device = args.device

        # Select the model type
        if args.model == 'ConvLSTM':
            self.model = ConvLSTM(args.input_dim, args.hid_dim, args.y_frames, args.n_layers, args.n_filters,
             args.filter_size, args.batch_size,
             args.dropout, args.use_bn, args.str_len)
        elif args.model == 'LSTM':
            self.model = LSTM(args.input_dim, args.hid_dim, args.y_frames, args.n_layers, args.batch_size, args.dropout,
                              args.use_bn)
        elif args.model == 'CNN':
            self.model = Conv1D(args.input_dim, args.y_frames, args.n_filters, args.filter_size, args.batch_size,
                                args.dropout)
        else:
            raise ValueError('In-valid model choice')

        self.model.to(self.device)

        self.pbounds = {
        'learning_rate': args.lr,
        'batch_size': args.batch_size
        }

        self.bayes_optimizer = BayesianOptimization(
        f=self.train,
        pbounds=self.pbounds
        )
		
	def train(self, learning_rate, batch_size):
        model = self.model
        batch_size = round(batch_size)
        loss_fn = torch.nn.CrossEntropyLoss()

        trainloader = DataLoader(self.trainset, batch_size=batch_size, 
								 shuffle=True, drop_last=True)

		optimizer = optim.Adam(model.parameters(), lr=learning_rate)
		model.train()
		model.zero_grad()
		optimizer.zero_grad()

		train_acc = 0.0
		train_loss = 0.0

        for i, (X, y) in enumerate(trainloader):

            X = X.transpose(0, 1).float().to(self.device)
            y_true = y[:, 0].long().to(self.device)

            model.zero_grad()
            optimizer.zero_grad()
            if model == 'ConvLSTM' or model == 'LSTM':
                model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

            y_pred = model(X)
            loss = loss_fn(y_pred, y_true.view(-1))
            loss.backward()
            optimizer.step()

            _, y_pred = torch.max(y_pred.data, 1)
            train_loss += loss.item()
            train_acc += (y_pred == y_true).sum()

        train_loss = train_loss / len(trainloader)
        train_acc = train_acc / len(trainloader)
        train_acc = float(train_acc)

        return train_loss, train_acc

    def validate(self, loss_fn, args):
        model = self.model
        valloader = DataLoader(self.valset, batch_size=args.batch_size,
                               shuffle=False, drop_last=True)
        model.eval()

        val_acc = 0.0
        val_loss = 0.0
        with torch.no_grad():
            for i, (X, y) in enumerate(valloader):

                X = X.transpose(0, 1).float().to(args.device)
                y_true = y[:, 0].long().to(args.device)
                if args.model == 'ConvLSTM' or args.model == 'LSTM':
                    model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

                y_pred = model(X)
                loss = loss_fn(y_pred, y_true.view(-1))

                _, y_pred = torch.max(y_pred.data, 1)
                val_loss += loss.item()
                val_acc += (y_pred == y_true).sum()

        val_loss = val_loss / len(valloader)
        val_acc = val_acc / len(valloader)
        val_acc = float(val_acc)

        return val_loss, val_acc

    def test(self, args):
        model = self.model
        testloader = DataLoader(self.testset, batch_size=args.batch_size,
                                shuffle=False, drop_last=True)
        model.eval()

        test_acc = 0.0
        with torch.no_grad():
            for i, (X, y) in enumerate(testloader):

                X = X.transpose(0, 1).float().to(args.device)
                y_true = y[:, 0].long().to(args.device)
                if args.model == 'ConvLSTM' or args.model == 'LSTM':
                    model.hidden = [hidden.to(args.device) for hidden in model.init_hidden()]

                y_pred = model(X)
                _, y_pred = torch.max(y_pred.data, 1)
                test_acc += (y_pred == y_true).sum()

        test_acc = test_acc / len(testloader)
        test_acc = float(test_acc)
        return test_acc

def experiment(mode, args):

    # ===== List for epoch-wise data ====== #
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    # ===================================== #

    loss_fn = torch.nn.CrossEntropyLoss()
    manager = Manager(args)

    for epoch in range(args.epoch):  # loop over the dataset multiple time
        if args.mode == 'train':
            print('Start training ... ')
            exp.save_exp_result(manager)
            manager.bayes_optimizer.maximize(args.init_points, args.n_iter, acq='ei', xi=0.01)

        elif args.mode == 'val':
            ts = time.time()
            print('Start validation ... ')
            val_loss, val_acc = manager.validate(model, loss_fn, args)
            # ====== Add Epoch Data ====== #
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            # ============================ #
            te = time.time()
            print(
            'Epoch {}, Acc: {:2.2f}, Loss: {:2.5f}. Took {:2.2f} sec'.format(epoch, val_acc, val_loss, te - ts))

    if args.mode == 'test':
        test_acc = manager.test(model, args)

    # ======= Add Result to Dictionary ======= #
    result = {}

    if args.mode == 'val':
        result['val_losses'] = val_losses
        result['val_accs'] = val_accs
        result['val_acc'] = val_acc
    else:
        result['test_acc'] = test_acc

    return vars(args), result
	
