class MRF:
    '''GRUMLIN implemented in jax'''
    def __init__(self, X, lengths=None, ss_hide=0.15, batch_size=128, 
               filters=512, win=18, lam=0.01,
               sw_unroll=4, sw_temp=1.0, sw_learn_temp=False,
               sw_open=None, sw_gap=None, sw_learn_gap=False,
               nat_contacts=None, nat_contacts_mask=None,
               nat_aln=None, use_nat_aln=False, add_aln_loss=False, aln_lam=1.0,
               seed=None, lr=0.1, norm_mode="fast",
               learn_bias=True, w_scale=0.1, 
               msa_memory = False, align_to_msa_frac = 0.0, pid_thresh = 1.0, pseudo = False):

        N,L,A = X.shape

        # inputs
        self.X = X
        self.lengths = X.sum([1,2]).astype(int) if lengths is None else lengths
        self.X_ref = self.X[:1]
        self.X_ref_len = self.lengths[0]  # singal dim

        self.nat_contacts = nat_contacts
        self.nat_contacts_mask = nat_contacts_mask
        self.nat_aln = nat_aln

        self.lr = lr*jnp.log(batch_size)/self.X_ref_len

        # seed for weight initialization and sub-sampling input
        self.key = laxy.KEY(seed)

        # params
        self.p = {"N":N, "L":L, "A":A, "batch_size":batch_size,
                  "sw_temp":sw_temp,"sw_learn_temp":sw_learn_temp,
                  "sw_unroll":sw_unroll,
                  "sw_open":sw_open,"sw_gap":sw_gap,"sw_learn_gap":sw_learn_gap,
                  "filters":filters, "win":win,
                  "x_ref_len":self.X_ref_len,
                  "ss_hide":ss_hide, "lam":lam*ss_hide*batch_size/N,
                  "use_nat_aln":use_nat_aln, "add_aln_loss":add_aln_loss, "aln_lam":aln_lam,
                  "norm_mode":norm_mode,
                  "learn_bias":learn_bias,"w_scale":w_scale, "msa_memory":msa_memory, 
                  "align_to_msa_frac":align_to_msa_frac, "pid_thresh":pid_thresh, "pseudo":pseudo}

        # initialize model
        self.init_params, self.model = self._get_model()

        self.model_aln = jax.jit(self._get_model(initialize_params=False, return_aln=True))
        self.opt = laxy.OPT(self.model, self.init_params, lr=self.lr)

  #####################################################################################################
  #####################################################################################################
    def _get_model(self, initialize_params=True, return_aln=False):
        p = self.p
        #######################
        # initialize params
        #######################
        if initialize_params:
            _params = {"mrf": laxy.MRF()(p["x_ref_len"], p["A"],
                                       use_bias=p["learn_bias"], key=self.key.get())} # w:[L,A,L,A]  b:[L,A]
            _params["emb"] = Conv1D_custom()(p["A"],p["filters"],p["win"],key=self.key.get()) # [512,20,18]

            _params["open"] = p["sw_open"]
            _params["gap"] = p["sw_gap"]
            _params["temp"] = p["sw_temp"]
            _params["msa"] = self.X[0,:p["x_ref_len"],...]

        # self-supervision
        def self_sup(x, key=None):
            if p["ss_hide"] == 1 or key is None:
                return x,x
            else:
                tmp = jax.random.uniform(key,[p["batch_size"],p["L"],1]) # [128,52,1]
                mask = (tmp > p["ss_hide"]).astype(x.dtype) # [128,52,1]
                return x*mask, x*(1-mask) # x = X[index] [128,52,20]

        # get alignment
        def get_aln(z, lengths, gap=None, open=None, temp=1.0, key=None): 
            # local-alignment (smith-waterman)
            if gap is None:
                aln_app = sw.sw_nogap(batch=True, unroll=p["sw_unroll"])
                aln = aln_app(z, lengths, temp)
            elif open is None:
                aln_app = sw.sw(batch=True, unroll=p["sw_unroll"])
                aln = aln_app(z, lengths, gap, temp)
            else:
                aln_app = sw.sw_affine(restrict_turns=True, batch=True, unroll=p["sw_unroll"])
                aln = aln_app(z, lengths, gap, open, temp)
            return aln

        #######################
        # setup the model
        #######################
        def _model(params, inputs):      
            # self-supervision (aka. masked-language-modeling)
            x_ms_in, x_ms_out = self_sup(inputs["x"], key=inputs["key"][0]) # [128,52,20]   [128,52,20]

            if p["use_nat_aln"]:
                aln = p_aln = inputs["aln"]
                
            else:
            # concatentate reference, get positional embedding
                x_ms_in_ = jnp.concatenate([inputs["x_ref"], x_ms_in],0) #[128,52,20] + [1,52,20] -> [129,52,20]
                emb = Conv1D_custom(params["emb"])(x_ms_in_,key=inputs["key"][1],scale=p["w_scale"]) # [129, 52, 512]

                # get alignment to reference
                if p["align_to_msa_frac"]>0:
                    embedded_msa = Conv1D_custom(params["emb"])(params["msa"][None,...],key=inputs["key"][1],scale=p["w_scale"]) # [1,52,512]
                    embedded_msa = embedded_msa[0,:p["x_ref_len"],...] # [52,512]
                    sm_mtx = emb[1:] @ ((1-self.p["align_to_msa_frac"]) * emb[0,:p["x_ref_len"]].T + self.p["align_to_msa_frac"] * embedded_msa.T) # [128,52,512] @ [512,52]
                else:
                    sm_mtx = emb[1:] @ emb[0,:p["x_ref_len"]].T  # [128,52,512] @ [512,52] -> [128,52,52]
                    
                # mask
                sm_mask = jnp.broadcast_to(inputs["x"].sum(-1,keepdims=True), sm_mtx.shape)  #[128,52,52]
                lengths = jnp.stack([inputs["lengths"],
                                     jnp.broadcast_to(p["x_ref_len"],inputs["lengths"].shape)],-1) #[128,2]
                
                # normalize rows/cols (to remove edge effects due to 1D-convolution)
                sm_mtx = norm_row_col(sm_mtx, sm_mask, p["norm_mode"]) #[128,52,52]
                
                
                if p["pseudo"]:
                    aln = jnp.sqrt(jax.nn.softmax(sm_mtx, axis=-1) * jax.nn.softmax(sm_mtx, axis=-2)) # [128,52,52]
                else:
                    sm_open = params["open"] if p["sw_learn_gap"] else laxy.freeze(params["open"])
                    sm_gap = params["gap"] if p["sw_learn_gap"] else laxy.freeze(params["gap"])
                    sm_temp = params["temp"] if p["sw_learn_temp"] else laxy.freeze(params["temp"])
                    aln = get_aln(sm_mtx, lengths, gap=sm_gap, open=sm_open, temp=sm_temp, key=inputs["key"][1]) #[128,52,52] & [128,2] -> [128,52,52]
                    
                x_msa = jnp.einsum("nia,nij->nja", x_ms_in, aln) # [128,52,20]
                x_msa_bias = x_msa.mean(0) # [52,20]
  
                # update MSA 
                if self.p["msa_memory"] != False:
                    if p["pid_thresh"]<=1.0 and p["pid_thresh"]>0:
                        pid  = jnp.einsum('nla,la->n', x_msa, x_msa[0,...])/ x_msa.shape[1] # [128]
                        x_msa_restricted = jnp.einsum('nia,n->nia',x_msa, (pid > p["pid_thresh"])) # [128,52,20] pid < 为0
                        num_surviving_seqs = (pid > p["pid_thresh"]).sum() + 1
                        x_msa_bias_restricted = (self.X[0,:p["x_ref_len"],...] + x_msa_restricted.sum(axis = 0))/num_surviving_seqs # X[0] + normal(x_msa_restricted)
                    else:
                        x_msa_bias_restricted = x_msa_bias  # [52,20]
                    params["msa"] = self.p["msa_memory"] * params["msa"] + (1-self.p["msa_memory"])* x_msa_bias_restricted[:p["x_ref_len"],...] #updata MSA params

                laxy.freeze(params["msa"]) # 冻结不 grad 参数

            if return_aln:
                return aln, sm_mtx  # aln [128,52,52]; sm_mtx [128,52,52]

            # align, gremlin, unalign
            x_msa = jnp.einsum("nia,nij->nja", x_ms_in, aln)  # [128,52,20] ,[128,52,52] -> [128,52,20]
            x_msa_pred, w = laxy.MRF(params["mrf"])(x_msa, return_w=True) # return y, w
            if p["learn_bias"] == False:
                x_msa_pred += jnp.log(x_msa.sum(0) + 0.01 * p["batch_size"]) # y=wx+b
            x_ms_pred_logits = jnp.einsum("nja,nij->nia", x_msa_pred, aln) # [128,52,20] ,[128,52,52] -> [128,52,20]

            x_ms_pred = jax.nn.softmax(x_ms_pred_logits, -1) # [128,52,20]

            # regularization
            l2_loss = 0.5*(p["L"]-1)*(p["A"]-1)*jnp.square(w).sum() # [1]
            if p["learn_bias"]:
                l2_loss += jnp.square(params["mrf"]["b"]).sum() # [1]

            # compute loss (pseudo-likelihood)
            cce_loss = -(x_ms_out * jnp.log(x_ms_pred + 1e-8)).sum() # sum([128,52,20]*[128,52,20])  [1]
            loss = cce_loss + p["lam"] * l2_loss  # [1]

            if p["add_aln_loss"]:
                a_bce = -inputs["aln"] * jnp.log(aln + 1e-8)
                b_bce = -jax.nn.relu(1-inputs["aln"]) * jnp.log(jax.nn.relu(1-aln) + 1e-8)
                bce = (sm_mask * (a_bce+b_bce)).sum()
                loss += p["aln_lam"] * bce

            return x_ms_pred,aln,loss

        if initialize_params: return _params, _model
        else: return _model
  
#####################################################################################################
#####################################################################################################
