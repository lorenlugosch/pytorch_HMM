import torch
import torch.utils.data
from collections import Counter
import os
from sklearn.model_selection import train_test_split

class Config:
	def __init__(self,N,path):
		self.N = N
		self.path = path

def read_config(path):
	config = Config(N=8,path=path)
	return config

def get_datasets(config):
	path = config.path

	lines = []
	for filename in os.listdir(os.path.join(path, "train")):
		with open(os.path.join(path, "train", filename), "r") as f:
			lines_ = f.readlines()
		#lines_[-1] += '\n'
		lines += lines_

	# get input and output alphabets
	Sx = list(Counter(("".join(lines))).keys()) # set of possible output letters
	train_lines, valid_lines = train_test_split(lines, test_size=0.1, random_state=42)
	train_dataset = TextDataset(train_lines, Sx)
	valid_dataset = TextDataset(valid_lines, Sx)

	config.M = len(Sx)
	config.Sx = Sx

	return train_dataset, valid_dataset

class TextDataset(torch.utils.data.Dataset):
	def __init__(self, lines, Sx):
		self.lines = lines # list of strings
		self.Sx = Sx
		pad_and_one_hot = PadAndOneHot(self.Sx) # function for generating a minibatch from strings
		self.loader = torch.utils.data.DataLoader(self, batch_size=32, num_workers=1, shuffle=True, collate_fn=pad_and_one_hot)

	def __len__(self):
		return len(self.lines)

	def __getitem__(self, idx):
		line = self.lines[idx].lstrip(" ").rstrip("\n").rstrip(" ").rstrip("\n")
		return line

#def one_hot(letters, M):
#	"""
#	letters : LongTensor of shape (batch size, sequence length)
#	M : integer
#	Convert batch of integer letter indices to one-hot vectors of dimension M (# of possible letters).
#	"""

#	out = torch.zeros(letters.shape[0], letters.shape[1], M)
#	for i in range(0, letters.shape[0]):
#		for t in range(0, letters.shape[1]):
#			out[i, t, letters[i,t]] = 1
#	return out

class PadAndOneHot:
	def __init__(self, Sx):
		self.Sx = Sx

	def __call__(self, batch):
		"""
		Returns a minibatch of strings, one-hot encoded and padded to have the same length.
		"""
		x = []
		batch_size = len(batch)
		for index in range(batch_size):
			x_ = batch[index]

			# convert letters to integers
			x.append([self.Sx.index(c) for c in x_])

		# pad all sequences with 0 to have same length
		x_lengths = [len(x_) for x_ in x]
		T = max(x_lengths)
		for index in range(batch_size):
			x[index] += [0] * (T - len(x[index]))
			x[index] = torch.tensor(x[index])

		# stack into single tensor and one-hot encode integer labels
		x = torch.stack(x) #one_hot(torch.stack(x), len(self.Sx))
		x_lengths = torch.tensor(x_lengths)

		return (x,x_lengths)
