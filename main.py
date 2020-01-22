import torch
from models import HMM
from data import get_datasets, read_config
from training import Trainer

# Generate datasets from text file
path = "data"
config = read_config(path)
train_dataset, valid_dataset = get_datasets(config)
checkpoint_path = "."

# Initialize model
model = HMM(config=config)

# Train the model
num_epochs = 10
trainer = Trainer(model, config, lr=0.0003)
trainer.load_checkpoint(checkpoint_path)

for epoch in range(num_epochs):
	print("========= Epoch %d of %d =========" % (epoch+1, num_epochs))
	train_loss = trainer.train(train_dataset)
	valid_loss = trainer.test(valid_dataset)
	trainer.save_checkpoint(epoch, checkpoint_path)

	print("========= Results: epoch %d of %d =========" % (epoch+1, num_epochs))
	print("train loss: %.2f| valid loss: %.2f\n" % (train_loss, valid_loss) )


