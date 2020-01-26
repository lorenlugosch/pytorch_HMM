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
		if self.is_cuda:
			x = x.cuda()
			T = T.cuda()

		batch_size = x.shape[0]; T_max = x.shape[1]
		log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
		log_alpha = torch.zeros(batch_size, T_max, self.N)
		if self.is_cuda: log_alpha = log_alpha.cuda()

		log_alpha[:, 0, :] = self.emission_model(x[:,0]) + log_state_priors
		print(log_alpha[:, 0, :])
		for t in range(1, T_max):
			log_alpha[:, t, :] = self.emission_model(x[:,t]) + self.transition_model(log_alpha[:, t-1, :], use_max=False)
			print(log_alpha[:, t, :])

		log_sums = log_alpha.logsumexp(dim=2)

		# Select the sum for the final timestep (each x has different length).
		log_probs = torch.gather(log_sums, 1, T.view(-1,1) - 1)
		return log_probs

	def sample(self, T=10):
		state_priors = torch.nn.functional.softmax(self.unnormalized_state_priors, dim=0)
		transition_matrix = torch.nn.functional.softmax(self.transition_model.unnormalized_transition_matrix, dim=1)
		emission_matrix = torch.nn.functional.softmax(self.emission_model.unnormalized_emission_matrix, dim=1)

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
			z_t = torch.distributions.categorical.Categorical(transition_matrix[:,z_t]).sample().item()
			if t < T-1: z.append(z_t)

		return x, z

	def viterbi(self, x, T):
		"""
		x : IntTensor of shape (batch size, T_max)
		T : IntTensor of shape (batch size)

		Find argmax_z log p(z|x) for each (x) in the batch.
		"""
		if self.is_cuda:
			x = x.cuda()
			T = T.cuda()

		batch_size = x.shape[0]; T_max = x.shape[1]
		log_state_priors = torch.nn.functional.log_softmax(self.unnormalized_state_priors, dim=0)
		log_delta = torch.zeros(batch_size, T_max, self.N).float()
		psi = torch.zeros(batch_size, T_max, self.N).long()
		if self.is_cuda:
			log_delta = log_delta.cuda()
			psi = psi.cuda()

		log_delta[:, 0, :] = self.emission_model(x[:,0]) + log_state_priors
		for t in range(1, T_max):
			max_val, argmax_val = self.transition_model(log_delta[:, t-1, :], use_max=True)
			log_delta[:, t, :] = self.emission_model(x[:,t]) + max_val
			psi[:, t, :] = argmax_val

		# Get the probability of the best path
		log_max = log_delta.max(dim=2)[0]
		best_path_scores = torch.gather(log_max, 1, T.view(-1,1) - 1)

		# This next part is a bit tricky to parallelize across the batch,
		# so we will do it separately for each example.
		z_star = []
		for i in range(0, batch_size):
			z_star_i = [ log_delta[i, T[i] - 1, :].max(dim=0)[1].item() ]
			for t in range(T[i] - 1, 0, -1):
				z_t = psi[i, t, z_star_i[0]].item()
				z_star_i.insert(0, z_t)

			z_star.append(z_star_i)

		return z_star, best_path_scores

def log_domain_matmul(log_A, log_B, use_max=False):
	"""
	log_A : m x n
	log_B : n x p

	output : m x p matrix

	Normally, a matrix multiplication
	computes out_{i,j} = sum_k A_{i,k} x B_{k,j}

	A log domain matrix multiplication
	computes out_{i,j} = logsumexp_k log_A_{i,k} + log_B_{k,j}

	This is needed for numerical stability
	when A and B are probability matrices.
	"""
	#m = log_A.shape[0]
	#n = log_A.shape[1]
	#p = log_B.shape[1]

	#log_A_expanded = torch.stack([log_A] * p, dim=2)
	#log_B_expanded = torch.stack([log_B] * m, dim=0)

	#elementwise_sum = log_A_expanded + log_B_expanded
	#out = torch.logsumexp(elementwise_sum, dim=1)

	out = genbmm.logbmm(log_A.unsqueeze(0).contiguous(), log_B.unsqueeze(1).contiguous())[0]

	return out

def maxmul(log_A, log_B):
	m = log_A.shape[0]
	n = log_A.shape[1]
	p = log_B.shape[1]

	log_A_expanded = torch.stack([log_A] * p, dim=2)
	log_B_expanded = torch.stack([log_B] * m, dim=0)

	elementwise_sum = log_A_expanded + log_B_expanded
	out1, out2 = torch.max(elementwise_sum, dim=1)

	return out1, out2

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
		# Each col needs to add up to 1 (in probability domain)
		log_transition_matrix = torch.nn.functional.log_softmax(self.unnormalized_transition_matrix, dim=1)

		# Matrix multiplication in the log domain
		#out = genbmm.logbmm(log_alpha.unsqueeze(0).contiguous(), transition_matrix.unsqueeze(0).contiguous())[0]
		if use_max:
			out1, out2 = maxmul(log_transition_matrix, log_alpha.transpose(0,1))
			return out1.transpose(0,1), out2.transpose(0,1)
		else:
			out = log_domain_matmul(log_transition_matrix, log_alpha.transpose(0,1))
			out = out.transpose(0,1)
			return out

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
		emission_matrix = torch.nn.functional.log_softmax(self.unnormalized_emission_matrix, dim=1)
		out = emission_matrix[:, x_t].transpose(0,1)

		return out
