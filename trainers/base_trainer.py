import nomenclature

import torch
import numpy as np
import os
import wandb
import tqdm

class NullEvaluator(object):
    def trainer_evaluate(*args, **kwargs):
        pass

class BaseTrainer(object):
    def __init__(self, args, model, **kwargs):
        self.args = args
        self.model = model
        self.clip = None

        self.optimizer = torch.optim.Adam(params = filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = 'min', patience = 5, factor = 0.1)

        if args.evaluator is not None:
            self.evaluator = nomenclature.EVALUATORS[args.evaluator](args, model)
        else:
            self.evaluator = NullEvaluator()

        os.makedirs(f'checkpoints/{self.args.group}_{args.name}/', exist_ok = True)

    def train_step(self, data):
        inputs, labels = data['image'], data['track_id']
        inputs = inputs.to(nomenclature.device)
        labels = labels.to(nomenclature.device)

        outputs = self.model(inputs)
        loss = self.criterion(outputs, labels.squeeze())
        return outputs, labels.squeeze(), loss

    def train(self, train_dataloader, val_dataloader):
        epoch_running_loss = 0.0
        global_step = 0
        for epoch in range(self.args.epochs):
            total = 0
            correct = 0
            running_loss = 0.0

            train_dataloader.dataset.on_epoch_end()

            for i, data in enumerate(train_dataloader):
                global_step += 1
                self.optimizer.zero_grad()

                outputs, labels, total_loss = self.train_step(data)

                total_loss.backward()
                if self.clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                self.optimizer.step()

                # TODO
                if self.args.num_classes == outputs.shape[-1]:
                    _, predicted = torch.max(outputs.data, 1)
                    correct += predicted.eq(labels.squeeze().data).cpu().sum().float()
                    total += labels.size(0)

                running_loss += total_loss.item()
                epoch_running_loss += total_loss.item()
                if i % self.args.log_every == self.args.log_every - 1:
                    if self.args.num_classes == outputs.shape[-1]:
                        print('[(%d / %d), %5d] Loss: %.5f | Accuracy: %.3f%% (%d/%d)' % (epoch + 1, self.args.epochs, i + 1, running_loss / self.args.log_every, 100.*correct/total, correct, total))
                        wandb.log({'batch_loss': running_loss / self.args.log_every, 'accuracy': 100. * correct / total}, step = global_step)
                    else:
                        print('[(%d / %d), %5d] Loss: %.5f' % (epoch + 1, self.args.epochs, i + 1, running_loss / self.args.log_every))
                        wandb.log({'batch_loss': running_loss / self.args.log_every}, step = global_step)

                    running_loss = 0.0

            if (epoch + 1) % self.args.eval_every == 0:
                self.model.train(False)
                self.evaluator.trainer_evaluate(step = global_step)
                self.model.train(True)

            if self.scheduler is not None:
                self.scheduler.step(epoch_running_loss)

            wandb.log({'epoch_loss': epoch_running_loss / len(train_dataloader)}, step = global_step)
            epoch_running_loss = 0.0

            torch.save(self.model.state_dict(), f'checkpoints/{self.args.group}_{self.args.name}/{self.args.model}_{self.args.dataset}_{epoch}.pt')
