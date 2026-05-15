import torch
import torch.nn as nn
from icecream import ic
import torch.nn.init as init
import torchvision.models as models
import timm

class MLP(nn.Module):

    def __init__(self,
                num_layer : int,
                activation : nn.Module,
                input_size : int = 512,
                hidden_div : int = 2, 
                output_size : int = 1, 
                normalize : nn.Module = nn.Identity,
                dropout_rate : float = 0.1):
        """Customized Multilayer perceptron with exponentially increasing layer sizes.

        Args:
            num_layer (int): Number of full layers, completed with an additional linear layer in the end.
            activation (nn.Module): Nonlinearities of full layers
            input_size (int, optional): Input size (needs to be adjusted for mutiple inputs). Defaults to 512.
            hidden_div (int, optional): Exponential rate of layer growth. Defaults to 2.
            output_size (int, optional): Number of outputs. Defaults to 1.
            normalize (nn.Module, optional): Class (not instance) of normilization. Defaults to nn.Identity.
            dropout_rate (float, optional): Dropout after every fully connected layer. Defaults to 0.1.
        """
        super(MLP, self).__init__()

        self.normalize = normalize

        self.hidden_sizes = [int(input_size / (hidden_div ** i)) for i in range(1, num_layer + 1)]

        self.linear_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activations = nn.ModuleList()

        for i in range(num_layer):
            input_size = self.hidden_sizes[i - 1] if i > 0 else input_size
            layer = nn.Linear(input_size, self.hidden_sizes[i])
            norm = normalize(self.hidden_sizes[i])

            setattr(self, f"layer{i + 1}", layer)
            setattr(self, f"batch_norm{i + 1}", norm)
            setattr(self, f"activ{i + 1}", activation)

            self.linear_layers.append(layer)
            self.norms.append(norm)
            self.activations.append(activation())

        self.output_layer = nn.Linear(self.hidden_sizes[-1], output_size)
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize linear layers
        for layer in [*self.linear_layers, self.output_layer]:
            init.xavier_uniform_(layer.weight, gain=init.calculate_gain('relu'))

    def forward(self, x : torch.Tensor, mask : torch.Tensor = None):
        
        assert x.dim() == 3, "Wrong input dimension consider reshaping!" # batch x sequence x features        

        if mask is not None:
            valid = (~mask).float()                       
            x = x * valid.unsqueeze(-1)                    
            x_sum = x.sum(dim=1)                           
            valid_count = valid.sum(dim=1, keepdim=True).clamp(min=1)
            x = x_sum / valid_count                        
        else:
            x = x.mean(dim=1)

        for layer, norm, activation in zip(self.linear_layers, self.norms, self.activations):
            x = self.dropout(activation(norm(layer(x))))

        x = self.output_layer(x)
        return x
    


class LSTM(nn.Module): 
    def __init__(self,
                num_layers : int,
                activation : nn.Module,
                input_size: int = 384,
                hidden_div: int = 512,
                output_size : int = 1, 
                normalize : nn.Module = nn.Identity,
                dropout_rate : float = 0.1): 
        """Customized LSTM with fixed dimensions.

        Args:
            activation (nn.Module): Nonlinearity to be used.
            output_size (int, optional): Number of outputs. Defaults to 1.
            normalize (nn.Module, optional): Class (not instance) of normalization. Defaults to nn.Identity.
            dropout_rate (float, optional): Dropout rate. Defaults to 0.1.
        """
    
        super(LSTM, self).__init__()
        input_size = input_size
        hidden_size = hidden_div
        num_layers = num_layers
        bidirectional = False
        num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout_rate)
        self.norm = normalize(hidden_size * num_directions)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation()
        self.output_layer = nn.Linear(hidden_size * num_directions, output_size)
        #add batch normalization 
        #add cnn after lstm 

    def forward(self, x : torch.Tensor, mask : torch.Tensor = None): 
        # Select the last output in the sequence
        #lstm_out_last = lstm_out[:, -1, :]
        return self.forward_(x, mask)[:, -1, :]
        
    def forward_(self, x : torch.Tensor, mask : torch.Tensor = None):
        """
        The forward pass of the model.
        
        Args:
            x (torch.Tensor): The input to the model with shape (batch_size, sequence_length, input_size).
            mask (torch.Tensor, optional): Mask to be applied. Defaults to None.
            
        Returns:
            torch.Tensor: The output of the model with shape (batch_size, output_size).
        """
        
        assert x.dim() == 3, "Wrong input dimension, consider reshaping!" # batch x sequence x features
        
        # apply mask to images not to be taken into account
        if not mask is None:
            x = torch.einsum('jkl,jk->jkl',x, ~mask)

        # LSTM input shape: (batch_size, sequence_length, input_size)
        # Pass through LSTM: output shape: (batch_size, sequence_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        lstm_out_last = lstm_out

        # Apply dropout, normalization, activation, and output layer
        x = self.dropout(self.activation(self.norm(lstm_out_last)))
        x = self.output_layer(x)

        return x
    


class AttentionBased_uncertainty(nn.Module):
    
    def __init__(self, 
                 dropout_rate : float = 0.5,
                 layer: int = 4,
                 head: int = 2,
                 dim_forward: int = 110):
        super(AttentionBased_uncertainty, self).__init__()
        #decrease more slowly
        
        self.single_img_mlp = nn.Linear(384, 384)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=384,
            nhead=head,
            dim_feedforward=dim_forward,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=layer, norm=None
        )

        self.volume_token = nn.Parameter(torch.randn(1, 1, 384))

        self.last_lin = nn.Sequential(
            nn.Linear(384, 384),
            nn.GELU(),
            nn.Linear(384, 2)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        
        x = self.single_img_mlp(x)

        x = torch.cat((x, self.volume_token.repeat(x.size(0), 1, 1)), dim=1)
        mask_n = torch.zeros((mask.shape[0], mask.shape[1] + 1), dtype=torch.bool, device=mask.device)
        mask_n[:, -1] = 0
        mask_n[:, 0:-1] = mask

        x = self.encoder(src=x, src_key_padding_mask=mask_n)
        x = x[:, -1]

        x = self.last_lin(x)
        
        return x



class AttentionBased(nn.Module):
    
    def __init__(self, 
                 dropout_rate : float = 0.1, 
                 input_size : int = 384):
        super(AttentionBased, self).__init__()

        
        self.projection = nn.Linear(input_size, 256) #64
        self.positional_encoding = nn.Embedding(
            num_embeddings=13, embedding_dim=256 #64
        )  # 1 x 512 x 512
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, #64
            nhead=4, #4
            dim_feedforward=512, #125 #1536 #512
            dropout=dropout_rate,
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer, num_layers=2, norm=None #2
        )

        self.volume_token = nn.Parameter(torch.randn(1, 1,256)) #64
        self.last_lin = nn.Linear(256, 1) #64

        #self.projection = nn.Linear(input_size, 384) #64
        #self.positional_encoding = nn.Embedding(
        #    num_embeddings=13, embedding_dim=384 #64
        #)  # 1 x 512 x 512
        #encoder_layer = nn.TransformerEncoderLayer(
        #    d_model=384, #64
        #    nhead=6, #4
        #    dim_feedforward=1536, #125
        #    dropout=dropout_rate,
        #    batch_first=True,
        #    norm_first=True,
        #)
        #self.encoder = nn.TransformerEncoder(
        #    encoder_layer=encoder_layer, num_layers=6, norm=None #2
        #)

        #self.volume_token = nn.Parameter(torch.randn(1, 1,384)) #64
        #self.last_lin = nn.Linear(384, 1) #64


        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.constant_(self.projection.bias, 0)

        # from Transformer class
        for p in self.encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        nn.init.xavier_uniform_(self.last_lin.weight)
        nn.init.constant_(self.last_lin.bias, 0)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):

        #choice:return self.forward_(x, mask)[:, -1, :]
        return self.forward_(x, mask)[:, -1, :]

    def forward_(self, x: torch.Tensor, mask: torch.Tensor):
        x = self.projection(x)
        x = torch.cat((self.volume_token.repeat(x.size(0), 1, 1), x), dim=1)
        src_key_padding_mask = torch.cat(
            (torch.zeros(x.size(0), 1, device=mask.device, dtype=torch.bool), mask),
            dim=1,
        )
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1)).to(x.device)
        #x = x + self.positional_encoding(torch.arange(x.size(1), device=x.device))

        #x = self.encoder(src=x, mask=mask, src_key_padding_mask=src_key_padding_mask, is_causal = True)  # batch x imgs x 512, is_causal = False: model sees everything, all steps 
        #x = self.encoder(src=x, mask=mask, is_causal = True)  # batch x imgs x 512, is_causal = False: model sees everything, all steps 
        x = self.encoder(src=x, src_key_padding_mask=src_key_padding_mask) 
        #x = nn.functional.relu(x)

        #choice:return x[:, 1:, :]
        x = self.last_lin(x)
        return x[:, 1:, :]
        
    


class fine_tuning_notused(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True, out_dim=128):
        super().__init__()

        # Load real convolutional network
        if backbone_name == "resnet18":
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()  # remove classification head

        elif backbone_name == "resnet50":
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()

        elif backbone_name.startswith("dinov2"):
            backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
            self.feature_dim = backbone.embed_dim  # for ViT-style models

        elif "dinov3" in backbone_name:  
            backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            self.feature_dim = backbone.num_features

        elif backbone_name.startswith("fomo4wheat"):
            backbone = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14"
            )

            ckpt = torch.load("data/models/backbone/FoMo4Wheat_base.pth", map_location="cpu")

            # FoMo checkpoint is a full model → extract weights
            backbone.load_state_dict(ckpt.state_dict(), strict=False)

            self.feature_dim = backbone.embed_dim

            print("[CNN] Loaded FoMo4Wheat ViT-B/14 on DINOv2 backbone")
            print("Lstm")



        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.backbone = backbone

        self.lstm = LSTM(
            num_layers=3,
            activation=nn.CELU,
            input_size=self.feature_dim,   # IMPORTANT
            hidden_div=512,                # same as your normal LSTM
            output_size=1,
            normalize=nn.LayerNorm,
            dropout_rate=0.5,
        )

    def freeze_backbone_layers(self):
        """
        Freeze all backbone layers except the last one.
        Works for both CNNs (ResNet) and ViTs (DINOv2 / DINOv3).
        """
        name = getattr(self.backbone, "__class__", type(self.backbone)).__name__.lower()
        print("Lstm")
        # -------------------------------
        # Case 1: Vision Transformer (DINOv2 / DINOv3)
        # -------------------------------
        if hasattr(self.backbone, "blocks"):
            # Freeze everything
            for p in self.backbone.parameters():
                p.requires_grad = False

            # Unfreeze only the *last transformer block*
            last_block = self.backbone.blocks[-1]
            for p in last_block.parameters():
                p.requires_grad = True

            print(f"[CNN] All transformer blocks frozen except the last one for {name}.")

        # -------------------------------
        # Case 2: ResNet
        # -------------------------------
        elif "resnet" in name:
            # Freeze all layers except the last residual block (layer4)
            for lname in ["conv1", "bn1", "layer1", "layer2", "layer3"]:
                layer = getattr(self.backbone, lname)
                for p in layer.parameters():
                    p.requires_grad = False

            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

            print("[CNN] All ResNet layers frozen except 'layer4'.")
            print("LSTM")


        else:
            print(f"[CNN] No known freeze rule for {name} — nothing frozen.")


    def forward_(self, x, mask=None):
        """
        x shape: (B, T, C, H, W)
        """
        
        B, T, C, H, W = x.shape
        
        # merge batch and time
        x = x.reshape(B*T, C, H, W)

        # extract per-view features
        feats = self.backbone(x)  # (B*T, feature_dim)
        feats = feats.reshape(B, T, -1)

        out = self.lstm.forward_(feats,mask)

        if out.dim() == 2:
            out = out.unsqueeze(-1)

        # regression head
        return out
    
    def forward(self, x, mask=None):
        return self.forward_(x, mask)[:, -1, :]




class fine_tuning(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=True, out_dim=128):
        super().__init__()

        # Load real convolutional network
        if backbone_name == "resnet18":
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()  # remove classification head

        elif backbone_name == "resnet50":
            backbone = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
            self.feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()

        elif backbone_name.startswith("dinov2"):
            backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
            self.feature_dim = backbone.embed_dim  # for ViT-style models

        elif "dinov3" in backbone_name:  
            backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            self.feature_dim = backbone.num_features

        elif backbone_name.startswith("fomo4wheat"):
            backbone = torch.hub.load(
                "facebookresearch/dinov2",
                "dinov2_vitb14"
            )

            ckpt = torch.load("data/models/backbone/FoMo4Wheat_base.pth", map_location="cpu")

            # FoMo checkpoint is a full model → extract weights
            backbone.load_state_dict(ckpt.state_dict(), strict=False)

            self.feature_dim = backbone.embed_dim

            print("[CNN] Loaded FoMo4Wheat ViT-B/14 on DINOv2 backbone")


        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")

        self.backbone = backbone

        #same mlp as above
        self.mlp = MLP(
            num_layer=3,   
            activation=nn.CELU,
            input_size=self.feature_dim,
            hidden_div=2,        
            output_size=1,
            normalize=nn.LayerNorm,          
            dropout_rate=0.5
        )


    def freeze_backbone_layers(self):
        """
        Freeze all backbone layers except the last one.
        Works for both CNNs (ResNet) and ViTs (DINOv2 / DINOv3).
        """
        name = getattr(self.backbone, "__class__", type(self.backbone)).__name__.lower()
        print("Model: MLP")
        # -------------------------------
        # Case 1: Vision Transformer (DINOv2 / DINOv3)
        # -------------------------------
        if hasattr(self.backbone, "blocks"):
            # Freeze everything
            for p in self.backbone.parameters():
                p.requires_grad = False

            # Unfreeze only the *last transformer block*
            last_block = self.backbone.blocks[-1]
            for p in last_block.parameters():
                p.requires_grad = True

            print(f"[CNN] All transformer blocks frozen except the last one for {name}.")

        # -------------------------------
        # Case 2: ResNet
        # -------------------------------
        elif "resnet" in name:
            # Freeze all layers except the last residual block (layer4)
            for lname in ["conv1", "bn1", "layer1", "layer2", "layer3"]:
                layer = getattr(self.backbone, lname)
                for p in layer.parameters():
                    p.requires_grad = False

            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

            print("[CNN] All ResNet layers frozen except 'layer4'.")
            print("MLP")

        else:
            print(f"[CNN] No known freeze rule for {name} — nothing frozen.")


    def forward(self, x, mask=None):
        """
        x shape: (B, T, C, H, W)
        """

        B, T, C, H, W = x.shape
        
        # merge batch and time
        x = x.reshape(B*T, C, H, W)

        # extract per-view features
        feats = self.backbone(x)  # (B*T, feature_dim)
        feats = feats.reshape(B, T, -1)

        return self.mlp(feats, mask)
    

#CNN = fine_tuning