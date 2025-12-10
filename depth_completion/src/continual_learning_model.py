import os, torch, torchvision
import torch.nn.functional as F
from utils.src import log_utils, net_utils


class ContinualLearningModel(torch.nn.Module):
    '''
    Wrapper class for continual learning parameters

    Arg(s):
        key_token_pool_size : int
            number of keys/tokens in the pool
    '''

    def __init__(self,
                 image_pool_size,
                 depth_pool_size,
                 device):
        super(ContinualLearningModel, self).__init__()
        
        self.image_pool_size = image_pool_size
        self.depth_pool_size = depth_pool_size
        self.device = device
        self.new_params = []

        self.dataset_uids = []  # Master list of seen datasets
        self.dataset_selectors = []  # latent-dim vector for each dataset

        self.i2_key_pools = torch.nn.ParameterDict() 
        self.i2_token_pools = torch.nn.ParameterDict()
        self.i2_linear = torch.nn.ParameterDict()
        self.d2_key_pools = torch.nn.ParameterDict() 
        self.d2_token_pools = torch.nn.ParameterDict() 
        self.d2_linear = torch.nn.ParameterDict()

        self.i3_key_pools = torch.nn.ParameterDict() 
        self.i3_token_pools = torch.nn.ParameterDict()
        self.i3_linear = torch.nn.ParameterDict()
        self.d3_key_pools = torch.nn.ParameterDict() 
        self.d3_token_pools = torch.nn.ParameterDict() 
        self.d3_linear = torch.nn.ParameterDict()

        self.i4_key_pools = torch.nn.ParameterDict() 
        self.i4_token_pools = torch.nn.ParameterDict()
        self.i4_linear = torch.nn.ParameterDict()
        self.d4_key_pools = torch.nn.ParameterDict() 
        self.d4_token_pools = torch.nn.ParameterDict() 
        self.d4_linear = torch.nn.ParameterDict()

        self.latent_key_pools = torch.nn.ParameterDict()
        self.latent_token_pools = torch.nn.ParameterDict()
        self.latent_linear = torch.nn.ParameterDict()

        # Move to device
        self.to(self.device)
        self.eval()
 
    
    def get_key_token_pool(self, dataset_uid, dims):
        """
        Get key and token pools for a dataset given its uid
        
        Arg(s):
            dataset_uid : str
                unique id of dataset
            dims: tuple
                tuple of dimensions for the key and token pools
        Returns:
            key and token pools for the dataset
        """
        if dataset_uid not in self.dataset_uids:
            return self.add_new_key_token_pool(dataset_uid, dims)
        else:
            return self.i2_key_pools[dataset_uid], self.i2_token_pools[dataset_uid], self.i2_linear[dataset_uid], \
                    self.d2_key_pools[dataset_uid], self.d2_token_pools[dataset_uid], self.d2_linear[dataset_uid], \
                    self.i3_key_pools[dataset_uid], self.i3_token_pools[dataset_uid], self.i3_linear[dataset_uid], \
                    self.d3_key_pools[dataset_uid], self.d3_token_pools[dataset_uid], self.d3_linear[dataset_uid], \
                    self.i4_key_pools[dataset_uid], self.i4_token_pools[dataset_uid], self.i4_linear[dataset_uid], \
                    self.d4_key_pools[dataset_uid], self.d4_token_pools[dataset_uid], self.d4_linear[dataset_uid], \
                    self.latent_key_pools[dataset_uid], self.latent_token_pools[dataset_uid], self.latent_linear[dataset_uid]


    def add_new_key_token_pool(self, dataset_uid, dims):
        '''
        Add and return token for a new unseen dataset

        Arg(s):
            dataset_uid : str
                unique id of dataset
            dims: tuple
                tuple of dimensions for the key and token
        Returns:
            torch.Tensor[float32] : added token
        '''
        # Unpack dimensions
        i2_dim, d2_dim, i3_dim, d3_dim, i4_dim, d4_dim, latent_dim = dims

        new_selector = torch.nn.Parameter(torch.randn((latent_dim, 1), device=self.device), requires_grad=True)

        # Create key and token pools
        new_i2_key_pool = torch.nn.Parameter(torch.empty((i2_dim, i2_dim), device=self.device), requires_grad=True)
        new_i2_token_pool = torch.nn.Parameter(torch.empty((self.image_pool_size, i2_dim), device=self.device), requires_grad=True)
        new_i2_linear = torch.nn.Parameter(torch.empty((i2_dim,1,1), device=self.device), requires_grad=True)
        new_d2_key_pool = torch.nn.Parameter(torch.empty((d2_dim, d2_dim), device=self.device), requires_grad=True)
        new_d2_token_pool = torch.nn.Parameter(torch.empty((self.depth_pool_size, d2_dim), device=self.device), requires_grad=True)
        new_d2_linear = torch.nn.Parameter(torch.empty((d2_dim,1,1), device=self.device), requires_grad=True)

        new_i3_key_pool = torch.nn.Parameter(torch.empty((i3_dim, i3_dim), device=self.device), requires_grad=True)
        new_i3_token_pool = torch.nn.Parameter(torch.empty((self.image_pool_size, i3_dim), device=self.device), requires_grad=True)
        new_i3_linear = torch.nn.Parameter(torch.empty((i3_dim,1,1), device=self.device), requires_grad=True)
        new_d3_key_pool = torch.nn.Parameter(torch.empty((d3_dim, d3_dim), device=self.device), requires_grad=True)
        new_d3_token_pool = torch.nn.Parameter(torch.empty((self.depth_pool_size, d3_dim), device=self.device), requires_grad=True)
        new_d3_linear = torch.nn.Parameter(torch.empty((d3_dim,1,1), device=self.device), requires_grad=True)

        new_i4_key_pool = torch.nn.Parameter(torch.empty((i4_dim, i4_dim), device=self.device), requires_grad=True)
        new_i4_token_pool = torch.nn.Parameter(torch.empty((self.image_pool_size, i4_dim), device=self.device), requires_grad=True)
        new_i4_linear = torch.nn.Parameter(torch.empty((i4_dim,1,1), device=self.device), requires_grad=True)
        new_d4_key_pool = torch.nn.Parameter(torch.empty((d4_dim, d4_dim), device=self.device), requires_grad=True)
        new_d4_token_pool = torch.nn.Parameter(torch.empty((self.depth_pool_size, d4_dim), device=self.device), requires_grad=True)
        new_d4_linear = torch.nn.Parameter(torch.empty((d4_dim,1,1), device=self.device), requires_grad=True)

        new_latent_key_pool = torch.nn.Parameter(torch.empty((latent_dim, latent_dim), device=self.device), requires_grad=True)
        new_latent_token_pool = torch.nn.Parameter(torch.empty((self.image_pool_size, latent_dim), device=self.device), requires_grad=True)
        new_latent_linear = torch.nn.Parameter(torch.empty((latent_dim,1,1), device=self.device), requires_grad=True)

        # Initialize parameters using kaiming_normal_
        torch.nn.init.kaiming_normal_(new_i2_key_pool)
        torch.nn.init.kaiming_normal_(new_i2_token_pool)
        torch.nn.init.kaiming_normal_(new_i2_linear)
        torch.nn.init.kaiming_normal_(new_d2_key_pool)
        torch.nn.init.kaiming_normal_(new_d2_token_pool)
        torch.nn.init.kaiming_normal_(new_d2_linear)

        torch.nn.init.kaiming_normal_(new_i3_key_pool)
        torch.nn.init.kaiming_normal_(new_i3_token_pool)
        torch.nn.init.kaiming_normal_(new_i3_linear)
        torch.nn.init.kaiming_normal_(new_d3_key_pool)
        torch.nn.init.kaiming_normal_(new_d3_token_pool)
        torch.nn.init.kaiming_normal_(new_d3_linear)

        torch.nn.init.kaiming_normal_(new_i4_key_pool)
        torch.nn.init.kaiming_normal_(new_i4_token_pool)
        torch.nn.init.kaiming_normal_(new_i4_linear)
        torch.nn.init.kaiming_normal_(new_d4_key_pool)
        torch.nn.init.kaiming_normal_(new_d4_token_pool)
        torch.nn.init.kaiming_normal_(new_d4_linear)

        torch.nn.init.kaiming_normal_(new_latent_key_pool)
        torch.nn.init.kaiming_normal_(new_latent_token_pool)
        torch.nn.init.kaiming_normal_(new_latent_linear)

        # Add to the key and token pool dicts
        self.dataset_uids.append(dataset_uid)
        self.dataset_selectors.append(new_selector)
        assert len(self.dataset_uids) == len(self.dataset_selectors), "# of dataset selectors must equal # of datasets!"
        
        self.i2_key_pools[dataset_uid] = new_i2_key_pool
        self.i2_token_pools[dataset_uid] = new_i2_token_pool
        self.i2_linear[dataset_uid] = new_i2_linear
        self.d2_key_pools[dataset_uid] = new_d2_key_pool
        self.d2_token_pools[dataset_uid] = new_d2_token_pool
        self.d2_linear[dataset_uid] = new_d2_linear
        
        self.i3_key_pools[dataset_uid] = new_i3_key_pool
        self.i3_token_pools[dataset_uid] = new_i3_token_pool
        self.i3_linear[dataset_uid] = new_i3_linear
        self.d3_key_pools[dataset_uid] = new_d3_key_pool
        self.d3_token_pools[dataset_uid] = new_d3_token_pool
        self.d3_linear[dataset_uid] = new_d3_linear
        
        self.i4_key_pools[dataset_uid] = new_i4_key_pool
        self.i4_token_pools[dataset_uid] = new_i4_token_pool
        self.i4_linear[dataset_uid] = new_i4_linear
        self.d4_key_pools[dataset_uid] = new_d4_key_pool
        self.d4_token_pools[dataset_uid] = new_d4_token_pool
        self.d4_linear[dataset_uid] = new_d4_linear
        
        self.latent_key_pools[dataset_uid] = new_latent_key_pool
        self.latent_token_pools[dataset_uid] = new_latent_token_pool
        self.latent_linear[dataset_uid] = new_latent_linear
        assert set(self.i4_key_pools.keys()) == set(self.dataset_uids)
        assert set(self.d4_token_pools.keys()) == set(self.dataset_uids)

        # Add params to optimizer
        self.new_params.append(new_selector)
            
        self.new_params.append(new_i2_key_pool)
        self.new_params.append(new_i2_token_pool)
        self.new_params.append(new_i2_linear)
        self.new_params.append(new_d2_key_pool)
        self.new_params.append(new_d2_token_pool)
        self.new_params.append(new_d2_linear)
        
        self.new_params.append(new_i3_key_pool)
        self.new_params.append(new_i3_token_pool)
        self.new_params.append(new_i3_linear)
        self.new_params.append(new_d3_key_pool)
        self.new_params.append(new_d3_token_pool)
        self.new_params.append(new_d3_linear)
        
        self.new_params.append(new_i4_key_pool)
        self.new_params.append(new_i4_token_pool)
        self.new_params.append(new_i4_linear)
        self.new_params.append(new_d4_key_pool)
        self.new_params.append(new_d4_token_pool)
        self.new_params.append(new_d4_linear)
        
        self.new_params.append(new_latent_key_pool)
        self.new_params.append(new_latent_token_pool)
        self.new_params.append(new_latent_linear)

        return new_i2_key_pool, new_i2_token_pool, new_i2_linear, new_d2_key_pool, new_d2_token_pool, new_d2_linear, \
                new_i3_key_pool, new_i3_token_pool, new_i3_linear, new_d3_key_pool, new_d3_token_pool, new_d3_linear, \
                new_i4_key_pool, new_i4_token_pool, new_i4_linear, new_d4_key_pool, new_d4_token_pool, new_d4_linear, \
                new_latent_key_pool, new_latent_token_pool, new_latent_linear


    def get_selector_key_idx(self, dataset_uid):
        '''
        Get the index of the selector key for a given dataset

        Arg(s):
            dataset_uid : str
                unique id of dataset
        Returns:
            int : index of the selector key
        '''
        return self.dataset_uids.index(dataset_uid)


    def get_new_params(self):
        '''
        Returns the list of new parameters added to the model and resets the list

        Returns:
            list[torch.Tensor[float32]] : list of new parameters
        '''
        new_params = self.new_params
        self.new_params = []
        return new_params


    def restore_model(self,
                      restore_path,
                      optimizer):
        '''
        Loads weights from checkpoint

        Arg(s):
            restore_paths : list[str]
                path to model weights, 1st for depth model and 2nd for pose model (if exists)
            optimizer : torch.optimizer or None
                current optimizer
        Returns:
            int : training step
            torch.optimizer : optimizer for depth or None if no optimizer is passed in
        '''

        # TokenCDC: Restore ALL TokenCDC params
        checkpoint = torch.load(restore_path, map_location=self.device)

        i2_key_pools_state_dict = checkpoint['i2_key_pools_state_dict']
        i2_token_pools_state_dict = checkpoint['i2_token_pools_state_dict']
        i2_linear_state_dict = checkpoint['i2_linear_state_dict']
        d2_key_pools_state_dict = checkpoint['d2_key_pools_state_dict']
        d2_token_pools_state_dict = checkpoint['d2_token_pools_state_dict']
        d2_linear_state_dict = checkpoint['d2_linear_state_dict']
        i3_key_pools_state_dict = checkpoint['i3_key_pools_state_dict']
        i3_token_pools_state_dict = checkpoint['i3_token_pools_state_dict']
        i3_linear_state_dict = checkpoint['i3_linear_state_dict']
        d3_key_pools_state_dict = checkpoint['d3_key_pools_state_dict']
        d3_token_pools_state_dict = checkpoint['d3_token_pools_state_dict']
        d3_linear_state_dict = checkpoint['d3_linear_state_dict']
        i4_key_pools_state_dict = checkpoint['i4_key_pools_state_dict']
        i4_token_pools_state_dict = checkpoint['i4_token_pools_state_dict']
        i4_linear_state_dict = checkpoint['i4_linear_state_dict']
        d4_key_pools_state_dict = checkpoint['d4_key_pools_state_dict']
        d4_token_pools_state_dict = checkpoint['d4_token_pools_state_dict']
        d4_linear_state_dict = checkpoint['d4_linear_state_dict']
        latent_key_pools_state_dict = checkpoint['latent_key_pools_state_dict']
        latent_token_pools_state_dict = checkpoint['latent_token_pools_state_dict']
        latent_linear_state_dict = checkpoint['latent_linear_state_dict']

        # Identify missing keys (keys in state_dict but not in model)
        i3_key_pools_missing_keys = set(i3_key_pools_state_dict.keys()) - self.i3_key_pools.keys()
        d4_token_pools_missing_keys = set(d4_token_pools_state_dict.keys()) - self.d4_token_pools.keys()
        assert i3_key_pools_missing_keys == d4_token_pools_missing_keys, "Image/Depth Key/Token pools must have the same keys!"
        missing_keys = i3_key_pools_missing_keys

        # Add pools for the missing keys (and add to new_params list to be added to optimizer)
        for mk in missing_keys:
            self.add_new_key_token_pool(mk,
                                        (i2_key_pools_state_dict[mk].shape[1],
                                            d2_key_pools_state_dict[mk].shape[1],
                                            i3_key_pools_state_dict[mk].shape[1],
                                            d3_key_pools_state_dict[mk].shape[1],
                                            i4_key_pools_state_dict[mk].shape[1],
                                            d4_key_pools_state_dict[mk].shape[1],
                                            latent_key_pools_state_dict[mk].shape[1]))
            if optimizer is not None:
                optimizer.add_param_group({'params' : self.get_new_params()})

        # Load the dataset selectors
        assert len(self.dataset_uids) == len(self.dataset_selectors), "# of dataset selectors must equal # of datasets!"
        with torch.no_grad():
            for i in range(len(self.dataset_uids)):
                self.dataset_selectors[i].copy_(checkpoint['selector_{}'.format(i)])

        # Now, load the state dicts
        self.i2_key_pools.load_state_dict(i2_key_pools_state_dict)
        self.i2_linear.load_state_dict(i2_linear_state_dict)
        self.i2_token_pools.load_state_dict(i2_token_pools_state_dict)
        self.d2_key_pools.load_state_dict(d2_key_pools_state_dict)
        self.d2_token_pools.load_state_dict(d2_token_pools_state_dict)
        self.d2_linear.load_state_dict(d2_linear_state_dict)
        self.i3_key_pools.load_state_dict(i3_key_pools_state_dict)
        self.i3_token_pools.load_state_dict(i3_token_pools_state_dict)
        self.i3_linear.load_state_dict(i3_linear_state_dict)
        self.d3_key_pools.load_state_dict(d3_key_pools_state_dict)
        self.d3_token_pools.load_state_dict(d3_token_pools_state_dict)
        self.d3_linear.load_state_dict(d3_linear_state_dict)
        self.i4_key_pools.load_state_dict(i4_key_pools_state_dict)
        self.i4_token_pools.load_state_dict(i4_token_pools_state_dict)
        self.i4_linear.load_state_dict(i4_linear_state_dict)
        self.d4_key_pools.load_state_dict(d4_key_pools_state_dict)
        self.d4_token_pools.load_state_dict(d4_token_pools_state_dict)
        self.d4_linear.load_state_dict(d4_linear_state_dict)
        self.latent_key_pools.load_state_dict(latent_key_pools_state_dict)
        self.latent_token_pools.load_state_dict(latent_token_pools_state_dict)
        self.latent_linear.load_state_dict(latent_linear_state_dict)

        if optimizer is not None:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                pass

        # Return the current step and optimizer
        return checkpoint['train_step'], optimizer


    def save_model(self, checkpoint_path, step, optimizer):
        '''
        Save weights of the model to checkpoint path

        Arg(s):
            checkpoint_path : str
                path to save checkpoint
            step : int
                current training step
            optimizer : torch.optim
                optimizer
        '''
        param_dict = {
                    'train_step': step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'dataset_uids': self.dataset_uids,
                    'i2_key_pools_state_dict': self.i2_key_pools.state_dict(),
                    'd2_key_pools_state_dict': self.d2_key_pools.state_dict(),
                    'i2_token_pools_state_dict': self.i2_token_pools.state_dict(),
                    'd2_token_pools_state_dict': self.d2_token_pools.state_dict(),
                    'i2_linear_state_dict': self.i2_linear.state_dict(),
                    'd2_linear_state_dict': self.d2_linear.state_dict(),
                    'i3_key_pools_state_dict': self.i3_key_pools.state_dict(),
                    'd3_key_pools_state_dict': self.d3_key_pools.state_dict(),
                    'i3_token_pools_state_dict': self.i3_token_pools.state_dict(),
                    'd3_token_pools_state_dict': self.d3_token_pools.state_dict(),
                    'i3_linear_state_dict': self.i3_linear.state_dict(),
                    'd3_linear_state_dict': self.d3_linear.state_dict(),
                    'i4_key_pools_state_dict': self.i4_key_pools.state_dict(),
                    'd4_key_pools_state_dict': self.d4_key_pools.state_dict(),
                    'i4_token_pools_state_dict': self.i4_token_pools.state_dict(),
                    'd4_token_pools_state_dict': self.d4_token_pools.state_dict(),
                    'i4_linear_state_dict': self.i4_linear.state_dict(),
                    'd4_linear_state_dict': self.d4_linear.state_dict(),
                    'latent_key_pools_state_dict': self.latent_key_pools.state_dict(),
                    'latent_token_pools_state_dict': self.latent_token_pools.state_dict(),
                    'latent_linear_state_dict': self.latent_linear.state_dict()
                    }

        assert len(self.dataset_uids) == len(self.dataset_selectors), "# of dataset selectors must equal # of datasets!"
        for i in range(len(self.dataset_uids)):
            param_dict['selector_{}'.format(i)] = self.dataset_selectors[i]

        torch.save(param_dict, checkpoint_path)
