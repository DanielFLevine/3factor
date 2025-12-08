import torch
import torch.nn as nn
import numpy as np

from constants import (
        DEVICE,
        POSALPHA,
        FREEZE_INPUT_WEIGHTS,
        FREEZE_OUTPUT_WEIGHTS,
        TURN_OFF_RECURRENT_WEIGHTS,
        FREEZE_VALUE_WEIGHTS,
        FREEZE_NEUROMODULATOR_WEIGHTS,
        SECONDMODULATOR,
        ZERO_RECURRENT_WEIGHTS,
    )

device = DEVICE

# Alpha configuration flags for different plasticity coefficient modes
POSALPHAINITONLY = False
VECTALPHA = False
SCALARALPHA = False
assert not (SCALARALPHA and VECTALPHA)  # One or the other

# RNN with plastic connections and neuromodulation ("DA").
# Plasticity only in the recurrent connections, for now.

class RetroModulRNN(nn.Module):
    def __init__(self, params):
        super(RetroModulRNN, self).__init__()
        # NOTE: 'outputsize' excludes the value and neuromodulator outputs!
        for paramname in ['outputsize', 'inputsize', 'hidden_size', 'batch_size']:
            if paramname not in params.keys():
                raise KeyError("Must provide missing key in argument 'params': "+paramname)
        NBDA = 2  # 2 DA neurons, we  take the difference  - see below
        self.params = params
        self.activ = torch.tanh
        if self.params['input_plastic']:
            self.i2h = torch.zeros(self.params['inputsize'], params['hidden_size'], requires_grad=False).to(device)
        else:
            self.i2h = torch.nn.Linear(self.params['inputsize'], params['hidden_size']).to(device)
            if FREEZE_INPUT_WEIGHTS:
                self.i2h.weight.requires_grad = False
                self.i2h.bias.requires_grad = False
        if TURN_OFF_RECURRENT_WEIGHTS and ZERO_RECURRENT_WEIGHTS:
            self.w = torch.zeros(params['hidden_size'], params['hidden_size'], requires_grad=False).to(device)
        elif TURN_OFF_RECURRENT_WEIGHTS:
            self.w = torch.nn.Parameter((  (1.0 / np.sqrt(self.params['hidden_size']))  * ( 2.0 * torch.rand(self.params['hidden_size'], self.params['hidden_size']) - 1.0) ).to(device), requires_grad=False)
        else:   
            self.w =  torch.nn.Parameter((  (1.0 / np.sqrt(self.params['hidden_size']))  * ( 2.0 * torch.rand(self.params['hidden_size'], self.params['hidden_size']) - 1.0) ).to(device), requires_grad=True)

        self.second_modulator = SECONDMODULATOR
        self.input_plastic = self.params.get('input_plastic', False)
        
        # Number of neuromodulators for recurrent connections
        num_recurrent_nms = params['neuromodulator_count']
        # Total neuromodulators (add one for input plastic if enabled)
        total_nms = num_recurrent_nms + (1 if self.input_plastic else 0)
        
        if SCALARALPHA:
            self.alphas = torch.nn.ParameterList([torch.nn.Parameter(.01 * (2.0 * torch.rand(1, 1) -1.0).to(device), requires_grad=True) for i in range(num_recurrent_nms)])
            if self.input_plastic:
                self.input_alpha = torch.nn.Parameter(.01 * (2.0 * torch.rand(1, 1) -1.0).to(device), requires_grad=True)
        elif VECTALPHA:
            self.alphas = torch.nn.ParameterList([torch.nn.Parameter(.01 * (2.0 * torch.rand(params['hidden_size'], 1) -1.0).to(device), requires_grad=True) for i in range(num_recurrent_nms)])  # A column vector, so each neuron has a single plasticity coefficient applied to all its input connections
            if self.input_plastic:
                self.input_alpha = torch.nn.Parameter(.01 * (2.0 * torch.rand(params['hidden_size'], 1) -1.0).to(device), requires_grad=True)
        else:
            self.alphas = torch.nn.ParameterList([torch.nn.Parameter(.01 * (2.0 * torch.rand(params['hidden_size'], params['hidden_size']) -1.0).to(device), requires_grad=True) for i in range(num_recurrent_nms)])
            if self.input_plastic:
                self.input_alpha = torch.nn.Parameter(.01 * (2.0 * torch.rand(params['hidden_size'], params['inputsize']) -1.0).to(device), requires_grad=True)
        if POSALPHA or  POSALPHAINITONLY:
            self.alphas = torch.nn.ParameterList([torch.nn.Parameter(torch.abs(alpha), requires_grad=True) for alpha in self.alphas])
            if self.input_plastic and (POSALPHA or POSALPHAINITONLY):
                self.input_alpha = torch.nn.Parameter(torch.abs(self.input_alpha), requires_grad=True)
        self.etaet = torch.nn.Parameter((.7 * torch.ones(1)).to(device), requires_grad=True)  # Everyone has the same etaet
        self.neuromodulator_multipliers = torch.nn.ParameterList([torch.nn.Parameter((1.0 * torch.ones(1)).to(device), requires_grad=True) for i in range(num_recurrent_nms)])
        if self.input_plastic:
            self.input_neuromodulator_multiplier = torch.nn.Parameter((1.0 * torch.ones(1)).to(device), requires_grad=True)
        self.h2nms  = torch.nn.ModuleList([torch.nn.Linear(params['hidden_size'], NBDA).to(device) for i in range(total_nms)])      # nm output (includes input nm if input_plastic)
        self.h2o = torch.nn.Linear(params['hidden_size'], self.params['outputsize']).to(device)  # Actual output
        self.h2v = torch.nn.Linear(params['hidden_size'], 1).to(device)          # V prediction
        
        # Store fixed random plastic weight initializations if enabled
        self.fixed_random_plastic_init = self.params.get('fixed_random_plastic_init', False)
        if self.fixed_random_plastic_init:
            # Create fixed random initializations for plastic weights (one per batch element)
            # These will be the same across episodes but different for each batch element
            bs = params['batch_size']
            self.register_buffer('fixed_plastic_inits', 
                                torch.stack([0.1 * (2.0 * torch.rand(params['hidden_size'], params['hidden_size']) - 1.0) 
                                           for _ in range(num_recurrent_nms * bs)]).view(num_recurrent_nms, bs, params['hidden_size'], params['hidden_size']).to(device))
            if self.input_plastic:
                self.register_buffer('fixed_input_plastic_init',
                                   0.1 * (2.0 * torch.rand(bs, params['hidden_size'], params['inputsize']) - 1.0).to(device))

    
    def forward(self, inputs, hidden, eligibility_trace, plastic_weights, input_eligibility_trace=None, input_plastic_weights=None):
        BATCHSIZE = inputs.shape[0]  #  self.params['batch_size']
        HS = self.params['hidden_size']
        IS = self.params['inputsize']
        assert hidden.shape[0] == eligibility_trace.shape[0] == BATCHSIZE
        
        # Compute input contribution
        if self.input_plastic:
            assert input_eligibility_trace is not None and input_plastic_weights is not None
            # Apply alpha to input plastic weights and compute input->hidden transformation
            total_input_plastic_weights = torch.mul(self.input_alpha, input_plastic_weights)
            input_contribution = torch.bmm(total_input_plastic_weights, inputs.view(BATCHSIZE, IS, 1)).view(BATCHSIZE, HS)
        else:
            input_contribution = self.i2h(inputs)
        
        # Compute recurrent contribution
        total_plastic_weights = torch.stack([torch.mul(self.alphas[i], plastic_weights[i]) for i in range(len(self.alphas))]).sum(dim=0)
        recurrent_contribution = torch.matmul(self.w + total_plastic_weights, hidden.view(BATCHSIZE, HS, 1)).view(BATCHSIZE, HS)
        
        # Combine and activate
        hactiv = self.activ(input_contribution + recurrent_contribution)
        
        activout = self.h2o(hactiv)  # Output layer. Pure linear, raw scores - will be softmaxed later
        valueout = self.h2v(hactiv)  # Value prediction

        # Now computing the Hebbian updates...

        # With batching, DAout is a matrix of size BS x 1
        # Recurrent neuromodulators use hidden activation
        nmouts = [torch.tanh(self.h2nms[i](hactiv)) for i in range(len(self.neuromodulator_multipliers))]

        # Separate neuromodulator scalars for recurrent vs input plastic weights
        num_recurrent_nms = len(self.neuromodulator_multipliers)
        nmscalars = [self.neuromodulator_multipliers[i] * (nmouts[i][:,0] - nmouts[i][:,1])[:,None] for i in range(num_recurrent_nms)]

        if self.input_plastic:
            # Input neuromodulator uses tanh of input contribution (not hidden activation)
            input_activ = torch.tanh(input_contribution)
            input_nmout = torch.tanh(self.h2nms[num_recurrent_nms](input_activ))
            input_nmscalar = self.input_neuromodulator_multiplier * (input_nmout[:,0] - input_nmout[:,1])[:,None]

        # Eligibility trace gets stamped into the plastic weights - gated by neuromodulator
        deltapws = [nmscalar.view(BATCHSIZE,1,1) * eligibility_trace for nmscalar in nmscalars]
        plastic_weights = [plastic_weight + deltapw for plastic_weight, deltapw in zip(plastic_weights, deltapws)]
        for plastic_weight in plastic_weights:
            torch.clip_(plastic_weight, min=-50.0, max=50.0)

        # Update input plastic weights if enabled
        if self.input_plastic:
            deltapw_input = input_nmscalar.view(BATCHSIZE, 1, 1) * input_eligibility_trace
            input_plastic_weights = input_plastic_weights + deltapw_input
            torch.clip_(input_plastic_weights, min=-50.0, max=50.0)

        # Updating the eligibility trace - Hebbian update with a simple decay
        # NOTE: the decay is for the eligibility trace, NOT the plastic weights (which never decay during a lifetime, i.e. an episode)
        deltaet =  torch.bmm(input_contribution.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) # batched outer product; at this point 'hactiv' is the output and 'hidden' is the input  (i.e. ativities from previous time step)
        # deltaet =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) -  et * hactiv[:, :, None] ** 2  # Oja's rule  (...? anyway, doesn't ensure stability with tanh and arbitrary damult / etaet)
        # deltaet =  torch.bmm(hactiv.view(BATCHSIZE, HS, 1), hidden.view(BATCHSIZE, 1, HS)) -   hactiv.view(BATCHSIZE, HS, 1) * et  # Instar rule (?)

        deltaet = torch.tanh(deltaet)
        eligibility_trace = (1 - self.etaet) * eligibility_trace + self.etaet *  deltaet

        # Update input eligibility trace if enabled
        if self.input_plastic:
            deltaet_input = torch.bmm(hactiv.view(BATCHSIZE, HS, 1), inputs.view(BATCHSIZE, 1, IS))  # outer product of post-synaptic (hactiv) and pre-synaptic (inputs)
            deltaet_input = torch.tanh(deltaet_input)
            input_eligibility_trace = (1 - self.etaet) * input_eligibility_trace + self.etaet * deltaet_input

        hidden = hactiv

        if self.input_plastic:
            return activout, valueout, nmscalars, hidden, eligibility_trace, plastic_weights, input_eligibility_trace, input_plastic_weights
        else:
            return activout, valueout, nmscalars, hidden, eligibility_trace, plastic_weights




    def initialZeroET(self, mybs):
        return torch.zeros(mybs, self.params['hidden_size'], self.params['hidden_size'], requires_grad=False).to(device)

    def initialZeroPlasticWeights(self, mybs, nm_idx=0):
        if self.fixed_random_plastic_init:
            # Return the fixed random initialization for this neuromodulator
            return self.fixed_plastic_inits[nm_idx, :mybs].clone()
        else:
            return torch.zeros(mybs, self.params['hidden_size'], self.params['hidden_size'], requires_grad=False).to(device)
    
    def initialZeroInputET(self, mybs):
        return torch.zeros(mybs, self.params['hidden_size'], self.params['inputsize'], requires_grad=False).to(device)
    
    def initialZeroInputPlasticWeights(self, mybs):
        if self.fixed_random_plastic_init and self.input_plastic:
            # Return the fixed random initialization for input plastic weights
            return self.fixed_input_plastic_init[:mybs].clone()
        else:
            return torch.zeros(mybs, self.params['hidden_size'], self.params['inputsize'], requires_grad=False).to(device)
    
    def initialZeroState(self, mybs):
        return torch.zeros(mybs, self.params['hidden_size'], requires_grad=False ).to(device)