import torch
from tqdm import tqdm # for displaying progress bar
import os
import pandas as pd

class Trainer:
	def __init__(self, model, config, lr):
		self.model = model
		self.config = config
		self.lr = lr
		self.optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.00001)
		self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
		self.train_df = pd.DataFrame(columns=["loss","lr"])
		self.valid_df = pd.DataFrame(columns=["loss","lr"])

	def load_checkpoint(self, checkpoint_path):
		if os.path.isfile(os.path.join(checkpoint_path, "model_state.pth")):
			try:
				if self.model.is_cuda:
					self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model_state.pth")))
				else:
					self.model.load_state_dict(torch.load(os.path.join(checkpoint_path, "model_state.pth"), map_location="cpu"))
			except:
				print("Could not load previous model; starting from scratch")
		else:
			print("No previous model; starting from scratch")

	def save_checkpoint(self, epoch, checkpoint_path):
		try:
			torch.save(self.model.state_dict(), os.path.join(checkpoint_path, "model_state.pth"))
		except:
			print("Could not save model")
		
	def train(self, dataset):
		train_acc = 0
		train_loss = 0
		num_samples = 0
		self.model.train()
		print_interval = 1000
		for idx, batch in enumerate(tqdm(dataset.loader)):
			x,T = batch
			batch_size = len(x)
			num_samples += batch_size
			log_probs = self.model(x,T)
			loss = -log_probs.mean()
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()
			train_loss += loss.cpu().data.numpy().item() * batch_size
			if idx % print_interval == 0:
				print(loss.item())
				sampled_x, sampled_z = self.model.sample()
				print("".join([self.config.Sx[s] for s in sampled_x]))
				print(sampled_z)
		train_loss /= num_samples
		train_acc /= num_samples
		return train_loss

	def test(self, dataset, print_interval=20):
		test_acc = 0
		test_loss = 0
		num_samples = 0
		self.model.eval()
		print_interval = 1000
		for idx, batch in enumerate(dataset.loader):
			x,T = batch
			batch_size = len(x)
			num_samples += batch_size
			log_probs = self.model(x,T)
			loss = -log_probs.mean()
			test_loss += loss.cpu().data.numpy().item() * batch_size
			if idx % print_interval == 0:
				print(loss.item())
				sampled_x, sampled_z = self.model.sample()
				print("".join([self.config.Sx[s] for s in sampled_x]))
				print(sampled_z)
		test_loss /= num_samples
		test_acc /= num_samples
		self.scheduler.step(test_loss) # if the validation loss hasn't decreased, lower the learning rate
		return test_loss
		
