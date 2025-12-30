import sys
from pathlib import Path
current_file_path = Path(__file__).absolute() if "__file__" in globals() else Path.cwd()
current_dir = current_file_path.parent if "__file__" in globals() else current_file_path
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
import torch
import json
from Transformer.model import Transformer, LMConfig


class CheckpointLoader:
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        
           
        config_path = self.checkpoint_dir / "LMConfig.json"
        with open(config_path) as f:
            model_config = json.load(f)
        
           
        config = LMConfig(**model_config)
        self.model = Transformer(config=config)
        self.W = None
        
        self.d_model = model_config["d_model"]

    def load(self):
        self._load_model()
        return self.model, self.W

    def _load_model(self):
           
        model_file = self.checkpoint_dir / "backbone.pth"
           
        if not model_file.exists():
            raise FileNotFoundError(f"Base model parameter file does not exist: {model_file}")
        
        state_dict = torch.load(model_file, map_location='cpu', weights_only=True)
        
           
        loaded_params = 0
        for name, param in self.model.named_parameters():
            if name in state_dict:
                try:
                    param.data.copy_(state_dict[name])
                    print(f"Loaded parameter: {name}")
                    if name=="output.W":
                        self.W = state_dict[name]
                    loaded_params += 1
                except Exception as e:
                    print(f"Failed to load parameter {name}: {str(e)}")
            

