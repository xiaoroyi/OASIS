import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import time
import torch
import pandas as pd
import collections
from packaging import version
from torch.distributions import Categorical
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.distributed as dist
from accelerate import Accelerator

from transformers import Trainer
from transformers import logging
# from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_pt_utils import (
    get_parameter_names,
)
from transformers.utils import (
    is_sagemaker_mp_enabled
)

from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.gemma.modeling_gemma import GemmaAttention
from transformers.models.gemma2.modeling_gemma2 import Gemma2Attention
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention
import copy

from loss_func.repnoise_loss import rep_noise_loss

if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

# if is_torch_tpu_available():
#    import torch_xla.core.xla_model as xm
#    import torch_xla.debug.metrics as met
#    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)


def get_leaf_modules_with_grad(module):
    # if len(list(module.children())) == 0 and any(p.requires_grad for p in module.parameters()) and "lora_B" in module._get_name():
    #     return [module]
    # else:
    #     return [submodule for child in module.children() for submodule in get_leaf_modules_with_grad(child)]
    module_list = []
    for name, module in module.named_modules():
        #     if "lora_B" in name and "v_proj" in name and len(list(module.children())) == 0:
        #         module_list+= [module]
        # or isinstance(module, LlamaMLP)
        # if isinstance(module, LlamaAttention) or isinstance(module, OPTAttention):
        # if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention) or isinstance(module, MistralAttention) or isinstance(module, GemmaAttention) or isinstance(module, Qwen2Attention)or isinstance(module, Gemma2Attention):
        if 'LlamaAttention' in str(type(module)) or 'OPTAttention' in str(type(module)) or 'Qwen2Attention' in str(
                type(module)) or 'Gemma2Attention' in str(type(module)) or 'GemmaAttention' in str(
            type(module)) or 'MistralAttention' in str(type(module)):
            module_list += [module]

    return module_list

def get_leaf_modules_with_grad2(module):
    module_list = []
    for name, module in module.named_modules():
        module_type_str = str(type(module))
        if ('LlamaMLP' in module_type_str or 
            'OPTMLP' in module_type_str or 
            'Qwen2MLP' in module_type_str or 
            'Gemma2MLP' in module_type_str or 
            'GemmaMLP' in module_type_str or 
            'MistralMLP' in module_type_str):
            module_list += [module]
    return module_list
def get_llama_mlp_modules(model):
    module_list = []
    for name, module in model.named_modules():
        if 'LlamaMLP' in str(type(module)) or 'OPTMLP' in str(type(module)) or 'Qwen2MLP' in str(
                type(module)) or 'Gemma2MLP' in str(type(module)) or 'GemmaMLP' in str(
            type(module)) or 'MistralMLP' in str(type(module)):
            module_list += [module]
    return module_list





class OASIS(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dangerous_dataloader = None
        self.data_iter_dangerous = None
        self.dangerous_gradients = None
        self.sensitive_layers: List[nn.Module] = []
        self.module_to_name: Dict[nn.Module, str] = {}
        self.sensitive_layer_names: set = set()
        

    def specific_data_init(self, dangerous_dataset):
        logger.info("Loading harmful dataset for OASIS orthogonal projection.")
        self.dangerous_dataloader = self.get_dataloader(dangerous_dataset)
        self.data_iter_dangerous = iter(self.dangerous_dataloader)

    def get_dataloader(self, dataset) -> DataLoader:
        from transformers.trainer_utils import seed_worker
        from transformers.trainer_pt_utils import LengthGroupedSampler
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
        sampler = RandomSampler(dataset)
        dataloader_params = {
            "batch_size": 1,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "sampler": sampler,
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": seed_worker,
        }
        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    def sample_from_dangerous_dataset(self):
        if self.dangerous_dataloader is None:
            raise ValueError("dangerous_dataloader is not initialized. Call specific_data_init first.")
        try:
            batch = next(self.data_iter_dangerous)
        except StopIteration:
            self.data_iter_dangerous = iter(self.dangerous_dataloader)
            batch = next(self.data_iter_dangerous)
        return batch

    def compute_dangerous_gradients_and_select_layers(self, model, top_k=20):
        model.train()
        dangerous_inputs = self.sample_from_dangerous_dataset()
        dangerous_inputs = self._prepare_inputs(dangerous_inputs)
        
        self.module_to_name = {module: name for name, module in model.named_modules()}

        hooks = []
        dangerous_grads = {}
        all_leaf_modules = get_leaf_modules_with_grad(model)
        for layer in all_leaf_modules:
            def track_dangerous_gradient_hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    dangerous_grads[module] = grad_output[0].detach().clone()
            hook = layer.register_backward_hook(track_dangerous_gradient_hook)
            hooks.append(hook)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, dangerous_inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()

        self.accelerator.backward(loss)

        for hook in hooks:
            hook.remove()
        model.zero_grad()
        
        self.dangerous_gradients = dangerous_grads

        if not self.dangerous_gradients:
            logger.warning("No harmful gradients were computed; cannot select sensitive layers.")
            self.sensitive_layers = []
            return

        layer_norms = []
        for module, grad in self.dangerous_gradients.items():
            if grad is not None:
                norm = grad.norm(p=2).item()
                layer_norms.append((module, norm))

        sorted_layers_with_norm = sorted(layer_norms, key=lambda x: x[1], reverse=True)

        top_k_layers = sorted_layers_with_norm[:top_k]
        self.sensitive_layers = [module for module, norm in top_k_layers]
        

        self.sensitive_layer_names = {self.module_to_name[m] for m in self.sensitive_layers}
        
        logger.info("Selected top %s sensitive layers at step %s.", len(self.sensitive_layers), self.state.global_step)
        header = f"{'Rank':<5} | {'Layer Name':<70} | {'Gradient Norm (L2)':<20}"
        logger.info(header)
        logger.info("-" * (len(header) + 2))
        
        for i, (module, norm) in enumerate(top_k_layers):
            layer_name = self.module_to_name.get(module, "Unknown Layer")
            logger.info("%-5s | %-70s | %-20.6f", i + 1, layer_name, norm)
            

    def compute_dangerous_gradients_and_select_layers2(self, model, top_k_sensitive=16, num_final_layers=4):
        model.train()
        dangerous_inputs = self.sample_from_dangerous_dataset()
        dangerous_inputs = self._prepare_inputs(dangerous_inputs)
        
        self.module_to_name = {module: name for name, module in model.named_modules()}

        hooks = []
        dangerous_grads = {}
        all_leaf_modules = get_leaf_modules_with_grad(model)
        for layer in all_leaf_modules:
            def track_dangerous_gradient_hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    dangerous_grads[module] = grad_output[0].detach().clone()
            hook = layer.register_backward_hook(track_dangerous_gradient_hook)
            hooks.append(hook)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, dangerous_inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()

        self.accelerator.backward(loss)

        for hook in hooks:
            hook.remove()
        model.zero_grad()
        
        self.dangerous_gradients = dangerous_grads

        if not self.dangerous_gradients:
            logger.warning("No harmful gradients were computed; cannot select sensitive layers.")
            self.sensitive_layers = []
            return

        layer_norms = []
        for module, grad in self.dangerous_gradients.items():
            if grad is not None:
                norm = grad.norm(p=2).item()
                layer_norms.append((module, norm))

        sorted_layers_with_norm = sorted(layer_norms, key=lambda x: x[1], reverse=True)
        top_sensitive_modules = {module for module, norm in sorted_layers_with_norm[:top_k_sensitive]}
        last_n_modules = set()
        if len(all_leaf_modules) >= num_final_layers:
            last_n_modules = set(all_leaf_modules[-num_final_layers:])
        else:
            last_n_modules = set(all_leaf_modules) 
        final_selected_modules_set = top_sensitive_modules.union(last_n_modules)
        self.sensitive_layers = list(final_selected_modules_set)
        self.sensitive_layer_names = {self.module_to_name.get(m) for m in self.sensitive_layers if self.module_to_name.get(m)}
        module_to_norm_map = {module: norm for module, norm in layer_norms}
        final_layers_for_printing = [(m, module_to_norm_map.get(m, 0.0)) for m in self.sensitive_layers]
        sorted_final_layers = sorted(final_layers_for_printing, key=lambda x: x[1], reverse=True)

        logger.info(
            "Selected %s layers (top %s sensitive + last %s) at step %s.",
            len(self.sensitive_layers),
            top_k_sensitive,
            num_final_layers,
            self.state.global_step,
        )
        header = f"{'Rank':<5} | {'Layer Name':<70} | {'Gradient Norm (L2)':<20}"
        logger.info(header)
        logger.info("-" * (len(header) + 2))
        
        for i, (module, norm) in enumerate(sorted_final_layers):
            layer_name = self.module_to_name.get(module, "Unknown Layer")
            logger.info("%-5s | %-70s | %-20.6f", i + 1, layer_name, norm)

    def _zero_grad_for_non_sensitive_layers(self, model):
        import re

        #  True:  Attention and MLP
        #  False: Attention
        keep_mlp_in_sensitive_layers = True


        if not self.sensitive_layer_names:
            return
        prefixes_to_keep = set(self.sensitive_layer_names)
        if keep_mlp_in_sensitive_layers:
            inferred_mlp_prefixes = set()
            for attn_module_name in self.sensitive_layer_names:
                match = re.search(r'(.+\.layers\.\d+)\.self_attn', attn_module_name)
                if match:
                    layer_prefix = match.group(1)
                    mlp_prefix = f"{layer_prefix}.mlp"
                    inferred_mlp_prefixes.add(mlp_prefix)
            
            if inferred_mlp_prefixes:
                prefixes_to_keep.update(inferred_mlp_prefixes)
            else:
                logger.info("No paired MLP modules were inferred from the selected attention layers.")


        params_zeroed_count = 0
        params_kept_count = 0

        for name, param in model.named_parameters():
            if not (param.requires_grad and 'lora_' in name):
                continue

            should_keep_gradient = any(name.startswith(prefix) for prefix in prefixes_to_keep)
            
            if should_keep_gradient:
                params_kept_count += 1
            else:
                if param.grad is not None:
                    param.grad.zero_()
                params_zeroed_count += 1
                

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        self.do_grad_scaling = False

        def step():
            try:
                from sagemaker_dp import smp_forward_backward
                from transformers import is_sagemaker_mp_enabled
            except ImportError:
                def is_sagemaker_mp_enabled():
                    return False

            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            else:
                self.accelerator.backward(loss)
            return loss

        update_freq = getattr(self.args, "update_freq", 100)
        if self.state.global_step % update_freq == 0:
            self.compute_dangerous_gradients_and_select_layers(model, top_k=self.args.top_k_layers)
            # self.compute_dangerous_gradients_and_select_layers2(model, top_k_sensitive=12, num_final_layers=4)


        if not self.sensitive_layers:
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()
            self.accelerator.backward(loss)
            return loss.detach() / self.args.gradient_accumulation_steps

        self.sam_state = {"hooks": [], "gradient": {}}

        self.pre_first_step(model)
        step()
        self.after_first_step(model)
        model.zero_grad()
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)

        self._zero_grad_for_non_sensitive_layers(model)
        return loss.detach() / self.args.gradient_accumulation_steps

    @torch.no_grad()
    def pre_first_step(self, model):
        def track_gradient_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.sam_state["gradient"][module] = grad_output[0].detach().clone() / self.args.gradient_accumulation_steps

        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)
        
        for layer in self.sensitive_layers:
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []

        grad_norm = self._grad_norm(self.sam_state["gradient"])
        for module in self.sam_state["gradient"]:
            grad = self.sam_state["gradient"][module]
            scale = self.args.rho / (grad_norm + 1e-7)
            e_r = grad * scale
            
            if self.dangerous_gradients and module in self.dangerous_gradients:
                d = self.dangerous_gradients[module]
                if e_r.shape[-1] == d.shape[-1]:
                    d_mean = d.mean(dim=[0, 1])
                    d_norm_sq = d_mean @ d_mean + 1e-7
                    proj_scalar = (e_r @ d_mean) / d_norm_sq
                    projection = proj_scalar.unsqueeze(-1) * d_mean
                    e_r_orthogonal = e_r - projection
                    self.sam_state["gradient"][module] = e_r_orthogonal.detach().clone()
                else:
                    layer_name = self.module_to_name.get(module, "Unknown Layer")
                    logger.warning(
                        "Hidden dimension mismatch for module %s: e_r %s, d %s.",
                        layer_name,
                        e_r.shape[-1],
                        d.shape[-1],
                    )
                    self.sam_state["gradient"][module] = e_r.detach().clone()
            else:
                self.sam_state["gradient"][module] = e_r.detach().clone()

    @torch.no_grad()
    def pre_second_step(self, model):
        def perturbation_hook(module, input, output):
            if module in self.sam_state["gradient"]:
                perturbation = self.sam_state["gradient"][module]

                if isinstance(output, torch.Tensor):
                    output = (output,)

                if output[0].shape == perturbation.shape:
                    output[0].data = output[0].data + perturbation
                
                return output
            return output

        def apply_perturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

        for layer in self.sensitive_layers:
            apply_perturbation_hooks_recursive(layer, perturbation_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_second_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []

    @torch.no_grad()
    def _grad_norm(self, grads):
        if not grads:
            return 1e-7
        
        valid_grads = [grad for grad in grads.values() if grad is not None]
        if not valid_grads:
            return 1e-7

        norm = torch.norm(
            torch.stack([grad.norm(p=2) for grad in valid_grads]),
            p=2
        )
        return norm if norm > 0 else 1e-7




class BaseTrainer(Trainer):
    def get_dataloader(self, special_dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        from transformers.trainer_utils import (
            seed_worker 
        )
        from transformers.trainer_pt_utils import (
            LengthGroupedSampler, 
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
        
        sampler = RandomSampler(special_dataset)

        dataloader_params = {
            "batch_size": 1, 
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(special_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(special_dataset, **dataloader_params))
    def specific_data_init(self, dangerous_dataset):
        logger.info("Loading auxiliary WANDA dataset.")
        self.dangerous_dataloader = self.get_dataloader(dangerous_dataset)
        self.data_iter_danferous = iter(self.dangerous_dataloader)

    def sample_from_alignment(self, data_type):
        # Get a  batch
        if data_type == 'dangerous':
            data_iter = self.data_iter_danferous
            dataloader = self.dangerous_dataloader
        else:
            pass

        try:
            batch = next(data_iter)
        except (StopIteration):
            data_iter = iter(dataloader)
            batch = next(data_iter)
        return batch

    def check_dataset(self, inputs, status):
        if status == 'alignment':
            inputs = inputs
        else:
            inputs = self.sample_from_alignment(status)
        return inputs

    def switch_active_layers(self, n_layers, probability, total_layers):

        active_layers_indices = sorted(np.random.choice(range(total_layers), n_layers, replace=False, p=probability))
        logger.info("Activating layers at indices %s for the next steps.", active_layers_indices)
        return active_layers_indices
    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        inputs = self.check_dataset(inputs, status='alignment')
        model.train() 
        inputs = self._prepare_inputs(inputs)
        self.layers = get_leaf_modules_with_grad(model) 
        self.do_grad_scaling = False 

        def step(inputs_, model_): 
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model_, inputs_, self.args.gradient_accumulation_steps) 
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager(): 
                loss = self.compute_loss(model_, inputs_)
            if self.args.n_gpu > 1: 
                loss = loss.mean() 

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else: 
                self.accelerator.backward(loss)
            return loss
        self.sam_state = {} 
        self.sam_state["hooks"] = []
        self.sam_state["gradient"] = {}
        self.sam_state["gradient_special"] = {}
        self.sam_state["gradient_list"] = {}
        self.sam_state["gradient_probability"] = {}
        if self.state.global_step % self.args.probability_steps == 0:
            for layer in self.layers:
                for name, param in layer.named_parameters(): 
                    if 'lora' in name:
                        param.requires_grad = True

            inputs_ = self.check_dataset(inputs, status='dangerous')
            inputs_ = self._prepare_inputs(inputs_)
            self.pre_gradient_magnitude_step(model) 
            step(inputs_, model) 
            self.after_gradient_magnitude_step(model, title='dangerous') 
            model.zero_grad() 
            self.probability = self.sam_state["gradient_probability"]['dangerous'] 
        if self.state.global_step % self.args.lisa_interval_steps == 0: 
            self.active_layers_indices = self.switch_active_layers(self.args.lisa_activated_layers,
                                                                   probability=self.probability,
                                                                   total_layers=len(self.layers))

        inputs['activate_layers'] = []
        if len(self.layers) > 26:
            for i in self.active_layers_indices:
                if i != 0:
                    inputs['activate_layers'].append(i - 1)
                inputs['activate_layers'].append(i)
        else:
            inputs['activate_layers'] = self.active_layers_indices

        self.unfreeze_activate_layers() 

        self.pre_first_step(model) 
        step(inputs, model)
        self.after_first_step(model)
        model.zero_grad()
        self.pre_second_step(model) 
        loss = step(inputs, model)

        self.after_second_step(model)

        return loss.detach() / self.args.gradient_accumulation_steps

    @torch.no_grad() 
    def unfreeze_activate_layers(self): 
        for layer in self.layers: 
            for name, param in layer.named_parameters():
                param.requires_grad = False 
        for idx in self.active_layers_indices: 
            layer = self.layers[idx]
            for name, param in layer.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True

    @torch.no_grad()
    def pre_first_step(self, model): 
        def track_gradient_hook(module, grad_input, grad_output):
            self.sam_state["gradient"][module] = grad_output[0].detach().clone() / self.args.gradient_accumulation_steps

        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  

        for idx in self.active_layers_indices:
            layer = self.layers[idx]
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def pre_second_step(self, model): 
        def purturbation_hook(module, input, output):
            perturbation = self.sam_state["gradient"][module]
            output[0].data = output[0] + perturbation 
            return output
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks): 
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
        for idx in self.active_layers_indices:
            layer = self.layers[idx]
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_first_step(self, model): 
        for hook in self.sam_state["hooks"]:
            hook.remove() 
        self.sam_state["hooks"] = []
        grad_norm = self._grad_norm(self.sam_state["gradient"]) 
        for module in self.sam_state["gradient"]:
            grad = self.sam_state["gradient"][module] 
            scale = self.args.rho / (grad_norm + 1e-7) 
            e_r = (grad) * scale
            self.sam_state["gradient"][module] = e_r.detach().clone() 

    @torch.no_grad()
    def after_second_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []

    @torch.no_grad()
    def _grad_norm(self, poison_grads_representation): 
        norm = torch.norm(
            torch.stack([
                (poison_grads_representation[name]).norm(p=2)
                for name in poison_grads_representation
            ]),
            p=2
        )
        return norm

    @torch.no_grad()
    def pre_gradient_magnitude_step(self, model): 

        def track_gradient_hook(module, grad_input, grad_output):
            self.sam_state["gradient_special"][module] = grad_output[
                                                             0].detach().clone() / self.args.gradient_accumulation_steps

        def apply_backward_hooks_recursive(module, hook_fn_2, hooks):
            hook2 = module.register_backward_hook(hook_fn_2)
            hooks.append(hook2)

        for layer in self.layers:
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_gradient_magnitude_step(self, model, title): 
        self.sam_state["gradient_list"][title] = []
        for layer in self.layers:
            self.sam_state["gradient_list"][title].append(
                torch.norm(self.sam_state["gradient_special"][layer], 2).item())

        total = sum(self.sam_state["gradient_list"][title]) 
        self.sam_state["gradient_probability"][title] = [i / total for i in self.sam_state["gradient_list"][title]] 
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        self.sam_state["gradient_special"] = {}


class Vaccine(Trainer):
    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        self.do_grad_scaling = False

        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss

        # gradient_weight_optimizer_memory = calculate_gradient_weight_optimizer_memory(model)
        self.sam_state = {}
        self.sam_state["hooks"] = []
        self.sam_state["gradient"] = {}
        # self.sam_state["gradient_memory"] = []
        self.pre_first_step(model)
        step()
        self.after_first_step(model)
        model.zero_grad()
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)
        # for param in model.parameters():
        #     if param.grad is not None:
        #         param.grad*= 1/2

        # else:
        #     loss = step()
        return loss.detach() / self.args.gradient_accumulation_steps

    @torch.no_grad()
    def pre_first_step(self, model):
        def track_gradient_hook(module, grad_input, grad_output):
            # Store the gradients for the current layer
            self.sam_state["gradient"][module] = grad_output[0].detach().clone() / self.args.gradient_accumulation_steps
            # self.sam_state["gradient_memory"].append(grad_output[0].detach().numel())

        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)  # Append the hook to the list

        # Call the function with the initial empty hooks list
        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            perturbation = self.sam_state["gradient"][module]
            output[0].data = output[0] + perturbation
            return output

        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # self.sam_state["gradient_memory"] = []

        grad_norm = self._grad_norm(self.sam_state["gradient"])
        for module in self.sam_state["gradient"]:
            # grad_norm = self._grad_norm(self.sam_state["gradient"][module])
            grad = self.sam_state["gradient"][module]
            scale = self.args.rho / (grad_norm + 1e-7)
            e_r = (grad) * scale
            self.sam_state["gradient"][module] = e_r.detach().clone()

    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.sam_state["e_r"]:
        #     module.weight.data -= self.sam_state["e_r"][module]
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

    @torch.no_grad()
    def _grad_norm(self, poison_grads_representation):
        norm = torch.norm(
            torch.stack([

                (poison_grads_representation[name]).norm(p=2)

                # ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for name in poison_grads_representation
            ]),
            p=2
        )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm





   
class VaccineOrthogonal2(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dangerous_dataloader = None
        self.data_iter_dangerous = None
        self.dangerous_gradients = None  

    def specific_data_init(self, dangerous_dataset):
        logger.info("Loading harmful dataset for Vaccine orthogonal projection.")
        self.dangerous_dataloader = self.get_dataloader(dangerous_dataset)
        self.data_iter_dangerous = iter(self.dangerous_dataloader)

    def get_dataloader(self, dataset) -> DataLoader:
        from transformers.trainer_utils import (
            seed_worker 
        )
        from transformers.trainer_pt_utils import (
            LengthGroupedSampler, 
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
        sampler = RandomSampler(dataset)
        dataloader_params = {
            "batch_size": 1,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "sampler": sampler,
            "drop_last": self.args.dataloader_drop_last,
            "worker_init_fn": seed_worker,
        }
        return self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

    def sample_from_dangerous_dataset(self):

        if self.dangerous_dataloader is None:
            raise ValueError("dangerous_dataloader specific_data_init")
        try:
            batch = next(self.data_iter_dangerous)
        except StopIteration:
            self.data_iter_dangerous = iter(self.dangerous_dataloader)
            batch = next(self.data_iter_dangerous)
        return batch

    def compute_dangerous_gradients(self, model):
        model.train()
        dangerous_inputs = self.sample_from_dangerous_dataset()
        dangerous_inputs = self._prepare_inputs(dangerous_inputs)
        
        hooks = []
        dangerous_grads = {}
        mlp_modules = get_llama_mlp_modules(model) 
        
        for layer in mlp_modules:
            def track_dangerous_gradient_hook(module, grad_input, grad_output):
                dangerous_grads[module] = grad_output[0].detach().clone()
            hook = layer.register_backward_hook(track_dangerous_gradient_hook)
            hooks.append(hook)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, dangerous_inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()

        self.accelerator.backward(loss)

        for hook in hooks:
            hook.remove()
        model.zero_grad()
        
        self.dangerous_gradients = dangerous_grads

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        self.do_grad_scaling = False

        def step():
            try:
                from sagemaker_dp import smp_forward_backward
                from transformers import is_sagemaker_mp_enabled
            except ImportError:
                def is_sagemaker_mp_enabled():
                    return False

            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss

        if self.state.global_step % 100 == 0:
            logger.info("Updating harmful gradients at step %s.", self.state.global_step)
            self.compute_dangerous_gradients(model)

        self.sam_state = {
            "hooks": [],
            "gradient": {}
        }

        self.pre_first_step(model)
        step()
        self.after_first_step(model)
        model.zero_grad()
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)

        return loss.detach() / self.args.gradient_accumulation_steps

    @torch.no_grad()
    def pre_first_step(self, model):
        def track_gradient_hook(module, grad_input, grad_output):
            self.sam_state["gradient"][module] = grad_output[0].detach().clone() / self.args.gradient_accumulation_steps

        def apply_backward_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_backward_hook(hook_fn)
            hooks.append(hook)

        mlp_modules = get_llama_mlp_modules(model) 
        for layer in mlp_modules:
            self.sam_state["gradient"][layer] = 0
            apply_backward_hooks_recursive(layer, track_gradient_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_first_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []

        grad_norm = self._grad_norm(self.sam_state["gradient"])
        for module in self.sam_state["gradient"]:
            grad = self.sam_state["gradient"][module]
            scale = self.args.rho / (grad_norm + 1e-7)
            e_r = grad * scale
            

            if self.dangerous_gradients and module in self.dangerous_gradients:
                d = self.dangerous_gradients[module]  
                if e_r.shape[-1] == d.shape[-1]:  
                    d_mean = d.mean(dim=[0, 1])  
                    d_norm_sq = d_mean @ d_mean + 1e-7 
                    
                    proj_scalar = (e_r @ d_mean) / d_norm_sq  
                    projection = proj_scalar.unsqueeze(-1) * d_mean  
                    e_r_orthogonal = e_r - 0.9 * projection
                    self.sam_state["gradient"][module] = e_r_orthogonal.detach().clone()
                else:
                    logger.warning(
                        "Hidden dimension mismatch for MLP module %s: e_r %s, d %s. Skipping projection.",
                        type(module).__name__,
                        e_r.shape[-1],
                        d.shape[-1],
                    )
                    self.sam_state["gradient"][module] = e_r.detach().clone()
            else:
                self.sam_state["gradient"][module] = e_r.detach().clone()

    @torch.no_grad()
    def pre_second_step(self, model):
        def perturbation_hook(module, input, output):
            perturbation = self.sam_state["gradient"][module]
            return output + perturbation

        def apply_perturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

        mlp_modules = get_llama_mlp_modules(model)  
        for layer in mlp_modules:
            apply_perturbation_hooks_recursive(layer, perturbation_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_second_step(self, model):
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []

    @torch.no_grad()
    def _grad_norm(self, grads):
        norm = torch.norm(
            torch.stack([grad.norm(p=2) for grad in grads.values()]),
            p=2
        )
        return norm



def get_leaf_modules_with_grad(module):
    # if len(list(module.children())) == 0 and any(p.requires_grad for p in module.parameters()) and "lora_B" in module._get_name():
    #     return [module]
    # else:
    #     return [submodule for child in module.children() for submodule in get_leaf_modules_with_grad(child)]
    module_list = []
    for name, module in module.named_modules():
        #     if "lora_B" in name and "v_proj" in name and len(list(module.children())) == 0:
        #         module_list+= [module]
        # or isinstance(module, LlamaMLP)
        # if isinstance(module, LlamaAttention) or isinstance(module, OPTAttention):
        # if isinstance(module,LlamaAttention) or isinstance(module, OPTAttention) or isinstance(module, MistralAttention) or isinstance(module, GemmaAttention) or isinstance(module, Qwen2Attention)or isinstance(module, Gemma2Attention):
        if 'LlamaAttention' in str(type(module)) or 'OPTAttention' in str(type(module)) or 'Qwen2Attention' in str(
                type(module)) or 'Gemma2Attention' in str(type(module)) or 'GemmaAttention' in str(
            type(module)) or 'MistralAttention' in str(type(module)):
            module_list += [module]
    return module_list

def get_llama_mlp_modules(model):
    module_list = []
    for name, module in model.named_modules():
        if 'LlamaMLP' in str(type(module)) or 'OPTMLP' in str(type(module)) or 'Qwen2MLP' in str(
                type(module)) or 'Gemma2MLP' in str(type(module)) or 'GemmaMLP' in str(
            type(module)) or 'MistralMLP' in str(type(module)):
            module_list += [module]
    return module_list



class RandomVaccineTrainer(Trainer):
    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss

        self.sam_state = {}
        self.sam_state["hooks"] = []
        self.sam_state["gradient"] = {}
        self.pre_second_step(model)
        loss = step()
        self.after_second_step(model)
        # for param in model.parameters():
        #     if param.grad is not None:
        #         param.grad*= 1/2

        # else:
        #     loss = step()
        return loss.detach() / self.args.gradient_accumulation_steps

    @torch.no_grad()
    def pre_second_step(self, model):
        def purturbation_hook(module, input, output):
            # Modify the output, for example, by adding a perturbatio
            variance = self.args.rho
            # Generate samples from a Gaussian distribution
            gaussian_samples = variance ** (1 / 2) * torch.randn_like(output[0])
            output[0].data = output[0] + gaussian_samples
            return output

        # Register forward hooks for adding perturbation
        def apply_purturbation_hooks_recursive(module, hook_fn, hooks):
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)

        leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        for layer in leaf_modules_with_grad:
            # Apply hooks to all layers, including nested Sequential blocks
            apply_purturbation_hooks_recursive(layer, purturbation_hook, self.sam_state["hooks"])

    @torch.no_grad()
    def after_second_step(self, model):
        # disable hook here
        # for module in self.sam_state["e_r"]:
        #     module.weight.data -= self.sam_state["e_r"][module]
        for hook in self.sam_state["hooks"]:
            hook.remove()
        self.sam_state["hooks"] = []
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

    @torch.no_grad()
    def _grad_norm(self, poison_grads_representation):
        norm = torch.norm(
            torch.stack([
                (poison_grads_representation[name]).norm(p=2)
                for name in poison_grads_representation
            ]),
            p=2
        )
        # norm = ( poison_grads_representation ).norm(p=2)
        return norm


class FITrainer(Trainer):

    def init(self, model):
        self.initial_weights = {}
        for name, module in model.named_modules():
            if "lora" in name and len(list(module.children())) == 0 and isinstance(module, torch.nn.Linear):
                self.initial_weights[module] = module.weight.data.detach().clone()
        self.round = 0

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            reg = 0
            for name, module in model.named_modules():
                if "lora" in name and len(list(module.children())) == 0 and isinstance(module, torch.nn.Linear):
                    reg += self.args.lamb * torch.sum(
                        self.fisher_vector[module] * torch.square(module.weight - self.initial_weights[module]))
                    # reg += self.args.lamb * torch.sum(torch.square(module.weight -self.initial_weights[module] ))
            loss += reg
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss

        if self.round == 0:
            self.fisher_vector = {module: 0 for name, module in model.named_modules() if
                                  "lora" in name and len(list(module.children())) == 0 and isinstance(module,
                                                                                                      torch.nn.Linear)}
            eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
            for stepsize, old_inputs in enumerate(eval_dataloader):
                # Update the observed num examples
                model.zero_grad()
                old_inputs = self._prepare_inputs(old_inputs)
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, old_inputs)
                self.accelerator.backward(loss)
                for name, module in model.named_modules():
                    if "lora" in name and len(list(module.children())) == 0 and isinstance(module, torch.nn.Linear):
                        self.fisher_vector[module] += torch.square(module.weight.grad.data.detach().clone())

        loss = step()
        # leaf_modules_with_grad = get_leaf_modules_with_grad(model)
        # for module in leaf_modules_with_grad:
        #     module.weight.grad*= (1-self.masks[index])
        #     index+=1
        self.round += 1
        return loss.detach() / self.args.gradient_accumulation_steps


class KLTrainer(Trainer):

    def init(self, model):
        import copy
        self.teacher_model_w = copy.deepcopy(model.state_dict())
        self.round = 0

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)

        def step():
            temp = {name: copy.deepcopy(param) for name, param in model.named_parameters() if param.requires_grad}
            with torch.no_grad():
                model.load_state_dict(self.teacher_model_w)
                teacher_outputs = self.model(**inputs,
                                             return_dict=True,
                                             use_cache=False,
                                             )
                model.load_state_dict(temp, strict=False)
            student_ouput = model(**inputs,
                                  return_dict=True,
                                  use_cache=False,
                                  )
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            import torch.nn.functional as F
            # Compute KL divergence
            kl_loss = self.args.lamb * torch.nn.KLDivLoss(reduction="batchmean")(F.log_softmax(student_ouput[1], dim=1),
                                                                                 F.softmax(teacher_outputs[1].detach(),
                                                                                           dim=1))
            # reg += self.args.lamb * torch.sum(torch.square(module.weight -self.initial_weights[module] ))
            # kl_loss = torch.mean(student_ouput[1])
            loss += kl_loss
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss

        loss = step()
        self.round += 1
        return loss.detach() / self.args.gradient_accumulation_steps


class TarTrainer(Trainer):
    def get_dataloader(self, special_dataset) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """

        from transformers.trainer_utils import (
            seed_worker
        )
        from transformers.trainer_pt_utils import (
            LengthGroupedSampler,
        )
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator

        sampler = RandomSampler(special_dataset)

        dataloader_params = {
            "batch_size": 10,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(special_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

        return self.accelerator.prepare(DataLoader(special_dataset, **dataloader_params))

        # def wanda_data_init(self, dangerous_dataset, safe_dataset):

    def specific_data_init(self, dangerous_dataset, model):
        logger.info("Loading auxiliary WANDA dataset.")
        self.dangerous_dataloader = self.get_dataloader(dangerous_dataset)
        # self.safe_dataloader = self.get_dataloader(safe_dataset)
        self.data_iter_danferous = iter(self.dangerous_dataloader)
        # self.data_iter_safe = iter(self.safe_dataloader)
        self.retain_model = copy.deepcopy(model)

    def sample_from_alignment(self, data_type):
        # Get a  batch
        if data_type == 'dangerous':
            data_iter = self.data_iter_danferous
            dataloader = self.dangerous_dataloader
        else:
            pass
            # data_iter = self.data_iter_safe
            # dataloader = self.safe_dataloader
        try:
            batch = next(data_iter)
        except (StopIteration):
            # If the iterator is exhausted, create a new iterator
            data_iter = iter(dataloader)
            batch = next(data_iter)
        return batch

    def check_dataset(self, inputs, status):
        if status == 'alignment':
            inputs = inputs
        else:
            inputs = self.sample_from_alignment(status)
        return inputs

    def log_p_loss(self, logits: torch.Tensor, labels: torch.Tensor, vocab_size: int
                   ) -> torch.Tensor:
        """
        Compute the log probability loss for a language model.

        This function calculates the cross-entropy loss between the predicted logits
        and the true labels, typically used in language modeling tasks.

        Args:
            logits (torch.Tensor): The predicted logits from the model, typically of shape
                                   (batch_size, sequence_length, vocab_size).
            labels (torch.Tensor): The true labels, typically of shape
                                   (batch_size, sequence_length).
            vocab_size (int): The size of the vocabulary.

        Returns:
            torch.Tensor: The computed loss as a scalar tensor.
        """
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        return loss

    def _filter_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Filter the input dictionary to keep only specific keys.

        This function takes a dictionary of input tensors and returns a new dictionary
        containing only the keys 'input_ids', 'attention_mask', and 'labels' if they exist
        in the original dictionary.

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary containing input tensors.

        Returns:
            Dict[str, torch.Tensor]: A filtered dictionary containing only the specified keys.
        """
        return {
            k: v
            for k, v in inputs.items()
            if k in ["input_ids", "attention_mask", "labels"]
        }

    def get_distributed_random_number(self, accelerator: Accelerator):
        random_number = torch.rand(1).to(accelerator.device)
        accelerator.wait_for_everyone()
        return random_number.item()
    def distributed_sample_adversary_lr(self, adversary_lr_samples, accelerator):
        rand_num = self.get_distributed_random_number(accelerator)
        adversary_lr = adversary_lr_samples[
            math.floor(rand_num * len(adversary_lr_samples))
        ]
        return adversary_lr

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        inputs = self.check_dataset(inputs, status='alignment')
        model.train()
        inputs = self._prepare_inputs(inputs)
        harmful_inputs = self.check_dataset(inputs, status='dangerous')
        harmful_inputs = self._prepare_inputs(harmful_inputs)

        self.layers = get_leaf_modules_with_grad(model)
        self.do_grad_scaling = False

        # adversary_lr_samples = [2e-6, 2e-5, 4e-5]
        #
        # adversary_lr = self.distributed_sample_adversary_lr(
        #     adversary_lr_samples, self.accelerator
        # )
        #
        # inner_optimizer = torch.optim.AdamW(model.parameters(), lr=adversary_lr)
        # inner_optimizer = self.accelerator.prepare_optimizer(inner_optimizer)
        # inner_scheduler = None

        def step():

            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, harmful_inputs)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)

            stored_grads = {name: param.grad.data.clone() for name, param in model.named_parameters() if
                            param.requires_grad}
            # inner_optimizer.step()
            # model.zero_grad(set_to_none=True)

            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.data -= self.args.rho*stored_grads[name]/grad_norm
                    param.data -= 0.01 * stored_grads[name]

            with self.compute_loss_context_manager():
                loss2 = self.compute_loss(model, inputs)
            if self.use_apex:
                with amp.scale_loss(loss2, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss2)
            stored_grads_tr = {name: param.grad.data.clone() for name, param in model.named_parameters() if
                               param.requires_grad}

            model.zero_grad()

            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         param.data += 0.1 * stored_grads[name]
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.data -= self.args.rho*stored_grads[name]/grad_norm
                    param.data += 0.01 * stored_grads[name]

            with self.compute_loss_context_manager():
                # loss3 = self.compute_loss(model, inputs)
                # _x_r = self._filter_inputs(inputs)
                # model_outputs = model(**_x_r, output_hidden_states=True)
                # with torch.no_grad():
                #     base_model_outputs = self.retain_model(**_x_r, output_hidden_states=True)
                # loss3 = self.log_p_loss(model_outputs.logits, _x_r.get("labels"), model.vocab_size)
                loss3 = self.compute_loss(model, inputs)

                # loss4 = loss3 + torch.mean(torch.stack([
                #     (torch.norm(base_hidden - model_hidden, dim=-1)).mean()
                #     for base_hidden, model_hidden in zip(
                #         base_model_outputs.hidden_states, model_outputs.hidden_states)]))
            if self.use_apex:
                with amp.scale_loss(loss3, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss3)

            # tr_gradient + loss4_gradient
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # param.grad += stored_grads[name]
                    param.data.grad = param.grad.data + 2 * stored_grads_tr[name]

            return loss3

        loss = step()
        return loss.detach() / self.args.gradient_accumulation_steps


class RepNoiseTrainer(Trainer):
    def init(self, harmful_dataset):
        # reploss needs standard dataset, load alpaca here
        from transformers.trainer_utils import (seed_worker)
        from torch.utils.data import DataLoader, RandomSampler
        data_collator = self.data_collator
        sampler = RandomSampler(harmful_dataset)
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if not isinstance(harmful_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = sampler
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
        self.harmful_dataloader = self.accelerator.prepare(DataLoader(harmful_dataset, **dataloader_params))

    def training_step(
            self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        model.train()
        inputs = self._prepare_inputs(inputs)
        # Get an iterator from the DataLoader
        data_iter = iter(self.harmful_dataloader)
        # Get the next batch
        harmful_inputs = next(data_iter)
        harmful_inputs = self._prepare_inputs(harmful_inputs)

        def step():
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                # loss = self.compute_loss(model, inputs)
                loss = rep_noise_loss(model, harmful_inputs, inputs, beta=self.args.lamb, alpha=self.args.rho)
            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)
            return loss

        loss = step()
        # with torch.no_grad():
        #     if self.round>=self.warm_up_round:
        #         for name, param in model.named_parameters():
        #             if param.requires_grad:
        #                 param.grad *= self.mask[name]

        return loss.detach() / self.args.gradient_accumulation_steps


