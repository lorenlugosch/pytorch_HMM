import torch
import genbmm

class HMM(torch.nn.Module):
	"""
	Hidden Markov Model.
	(For now, discrete observations only.) 
	- forward(): computes the log probability of an observation sequence.
	- viterbi(): computes the most likely state sequence.
	- sample(): draws a sample from p(x).
	"""
	def __init__(self, config):
		super(HMM, self).__init__()
		self.M = config.M # number of possible observations
		self.N = config.N # number of states
		self.unnormalized_state_priors = torch.nn.Parameter(torch.randn(self.N))
		self.transition_model = TransitionModel(self.N)
		self.emission_model = EmissionModel(self.N,self.M)
		self.is_cuda = torch.cuda.is_available()
		if self.is_cuda: self.cuda()
		
	def forward(self, x, T):
		"""
		x : IntTensor of shape (batch size, T_max)
		T : IntTensor of shape (batch size)

		Compute log p(x) for each example in the batch.
		T = length of each example
		"""
		if self.is_cuda: x = x.cuda()

		batch_size = x.shape[0]; T_max = x.shape[1]
		state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
		log_alpha = torch.zeros(batch_size, T_max, self.N)
		if self.is_cuda: log_alpha = log_alpha.cuda()

		log_alpha[:, 0, :] = self.emission_model(x[:,0]) + state_priors
		for t in range(1, T_max):
			log_alpha[:, t, :] = self.emission_model(x[:,0]) + self.transition_model(log_alpha[:, t-1, :], use_max=False)

		log_sums = log_alpha.logsumexp(dim=2)

		# Select the sum for the final timestep (each x has different length).
		log_probs = torch.gather(log_sums, 1, T.view(-1,1) - 1)
		return log_probs

	def sample(self, T=10):
		state_priors = torch.nn.functional.softmax(self.unnormalized_state_priors, dim=0)
		transition_matrix = torch.nn.functional.softmax(self.transition_model.unnormalized_transition_matrix, dim=0)
		emission_matrix = torch.nn.functional.softmax(self.emission_model.unnormalized_emission_matrix, dim=0)

		# sample initial state
		z_t = torch.distributions.categorical.Categorical(state_priors).sample().item()
		z = []
		x = []
		z.append(z_t)
		for t in range(0,T):
			# sample emission
			x_t = torch.distributions.categorical.Categorical(emission_matrix[z_t]).sample().item()
			x.append(x_t)

			# sample transition
			z_t = torch.distributions.categorical.Categorical(transition_matrix[z_t]).sample().item()
			if t < T-1: z.append(z_t)

		return x, z

	def viterbi(self, x, T):
		"""
		x : IntTensor of shape (batch size, T_max)
		T : IntTensor of shape (batch size)

		Find argmax_z log p(x|z) for each (x) in the batch.
		"""
		if self.is_cuda: x = x.cuda()

		batch_size = x.shape[0]; T_max = x.shape[1]
		state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
		log_delta = torch.zeros(batch_size, T_max, self.N).float()
		psi = torch.zeros(batch_size, T_max, self.N).long()
		if self.is_cuda: 
			log_delta = log_delta.cuda()
			psi = log_psi.cuda()

		log_delta[:, 0, :] = self.emission_model(x[:,0]) + state_priors
		for t in range(1, T_max):
			max_val, argmax_val = self.transition_model(log_delta[:, t-1, :], use_max=True)
			log_delta[:, t, :] = self.emission_model(x[:,0]) + max_val
			psi[:, t, :] = argmax_val

		# This next part is a bit tricky to parallelize across the batch,
		# so we will do it separately for each example.
		z_star = []
		for i in range(0, batch_size):
			z_star_i = [ log_delta[i, T[i] - 1, :].max()[1].item() ]
			for t in range(T[i] - 2, 0, -1):
				z_t = psi[i, t, z_star_i[0]]
				z_star_i.insert(0, z_t)

			z_star.append(z_star_i)

		return z_star

class TransitionModel(torch.nn.Module):
	"""
	- forward(): computes the log probability of a transition.
	- sample(): given a previous state, sample a new state.
	"""
	def __init__(self, N):
		super(TransitionModel, self).__init__()
		self.N = N # number of states
		self.unnormalized_transition_matrix = torch.nn.Parameter(torch.randn(N,N))

	def forward(self, log_alpha, use_max):
		"""
		log_alpha : Tensor of shape (batch size, N)

		Multiply previous timestep's alphas by transition matrix (in log domain)
		"""
		batch_size = log_alpha.shape[0]

		# Each col needs to add up to 1 (in probability domain)
		transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=0)

		# Matrix multiplication in the log domain
		#log_alpha_stacked = torch.stack([log_alpha]*self.N, dim=2)
		#log_A_stacked = torch.stack([transition_matrix.transpose(0,1)]*batch_size)
		#if use_max: out = (log_alpha_stacked + log_A_stacked).max(dim=1)
		#else: out = (log_alpha_stacked + log_A_stacked).logsumexp(dim=1)
		out = genbmm.logbmm(log_alpha.unsqueeze(0), transition_matrix.unsqueeze(0))

		return out[0]

class EmissionModel(torch.nn.Module):
	"""
	- forward(): computes the log probability of an observation.
	- sample(): given a state, sample an observation for that state.
	"""
	def __init__(self, N, M):
		super(EmissionModel, self).__init__()
		self.N = N # number of states
		self.M = M # number of possible observations
		self.unnormalized_emission_matrix = torch.nn.Parameter(torch.randn(N,M))
		
	def forward(self, x_t):
		"""
		x_t : LongTensor of shape (batch size)

		Get observation probabilities
		"""
		# Each col needs to add up to 1 (in probability domain)
		emission_matrix = torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=0)
		out = emission_matrix[:, x_t].transpose(0,1)

		return out
