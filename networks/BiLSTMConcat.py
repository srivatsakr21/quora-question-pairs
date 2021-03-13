import torch
import torch.nn as nn
import pytorch_lightning as pl


class LSTMModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        # self.pad_idx =pad_idx
        # self.embedding_size = embedding_size
        self.embedding = nn.Embedding(self.hparams['input_size'], self.hparams['embedding_size'],
                                      padding_idx=self.hparams['pad_idx'])
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lstm = nn.LSTM(self.hparams['embedding_size'],
                            self.hparams['hidden_size'],
                            num_layers=self.hparams['num_layers'],
                            bidirectional=self.hparams['bidirectional'],
                            dropout=self.hparams['dropout'])
        self.fc = nn.Linear(self.hparams['hidden_size'] * 2, self.hparams['output_size'])
        self.dropout = nn.Dropout(self.hparams['dropout'])
        self.criterion = nn.BCEWithLogitsLoss()

    def set_embeddings(self, text):
        # text  = self.datasetup.get_fields()
        UNK_IDX = text.vocab.stoi[text.unk_token]
        self.embedding.weight.data.copy_(text.vocab.vectors)
        self.embedding.weight.data[UNK_IDX] = torch.zeros(self.hparams['embedding_size'])
        self.embedding.weight.data[self.hparams['pad_idx']] = torch.zeros(self.hparams['embedding_size'])

    def binary_accuracy(self, preds, y):
        """
        Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        """

        # round predictions to the closest integer
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).float()  # convert into float for division
        acc = correct.sum() / len(correct)
        return acc

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden_features, cell_state) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden_features[-2, :, :], hidden_features[-1, :, :]), dim=1))
        return self.fc(hidden)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'],
                                 weight_decay=self.hparams['weight_decay'])
        return optim

    def training_step(self, train_batch, batch_idx):
        question1,question2,duplicate = train_batch.question1,train_batch.question2,train_batch.is_duplicate
        ques_cat = torch.cat((question1,question2),0)
        predictions = self.forward(ques_cat).squeeze(1)
        train_loss = self.criterion(predictions, duplicate)
        acc = self.binary_accuracy(predictions, duplicate)
        self.logger.experiment.add_scalar('Train Loss', train_loss, self.current_epoch)
        # self.log('train_acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {'loss': train_loss, 'acc': acc}

    def validation_step(self, val_batch, batch_idx):
        question1,question2,duplicate = val_batch.question1,val_batch.question2,val_batch.is_duplicate
        ques_cat = torch.cat((question1,question2),0)
        predictions = self.forward(ques_cat).squeeze(1)
        val_loss = self.criterion(predictions, duplicate)
        acc = self.binary_accuracy(predictions, duplicate)
        self.logger.experiment.add_scalar('Val Loss', val_loss, self.current_epoch)
        # self.log('val_acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {'loss': val_loss, 'acc': acc}

    def test_step(self, test_batch, batch_idx):
        question1,question2,duplicate = test_batch.question1,test_batch.question2,test_batch.is_duplicate
        ques_cat = torch.cat((question1,question2),0)
        predictions = self.forward(ques_cat).squeeze(1)
        test_loss = self.criterion(predictions, duplicate)
        acc = self.binary_accuracy(predictions, duplicate)
        # self.log('test_acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return {'loss': test_loss, 'acc': acc}

    def validation_end(self, outputs):
        val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        val_acc = torch.stack([x['acc'] for x in outputs]).mean()
        return {'val_loss': val_loss, 'val_acc': val_acc}
