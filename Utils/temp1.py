
class SpatialTransformer(nn.Module):
    def __init__(self, c1):
        #super().__init()
        super(SpatialTransformer, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(c1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(14),
            nn.ReLU(True)
        )
        # Calculate the size of the input to the fully connected layer
        self.fc_loc_size = 8*14*14 # Adjust for your input size
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_loc_size, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        # pdb.set_trace()
        xs = self.localization(x)
        # xs = xs.view(-1, self.fc_loc_size)
        xs=torch.flatten(xs,start_dim=1)
        # pdb.set_trace()
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        # theta=theta.expand(x.shape[0],-1,-1)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x