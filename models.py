from all import nn
import torch


def nature_dqn(env, frames=4):
    return nn.Sequential(
        nn.Scale(1 / 255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear0(512, env.action_space.n)
    )


def nature_ddqn(env, frames=4):
    return nn.Sequential(
        nn.Scale(1 / 255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Dueling(
            nn.Sequential(
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear0(512, 1)
            ),
            nn.Sequential(
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear0(512, env.action_space.n)
            ),
        )
    )


def nature_features(frames=4):
    return nn.Sequential(
        nn.Scale(1 / 255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
    )


def nature_value_head():
    return nn.Linear(512, 1)


def nature_policy_head(env):
    return nn.Linear0(512, env.action_space.n)


def nature_c51(env, frames=4, atoms=51):
    return nn.Sequential(
        nn.Scale(1 / 255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(3136, 512),
        nn.ReLU(),
        nn.Linear0(512, env.action_space.n * atoms)
    )


def nature_rainbow(env, frames=4, hidden=512, atoms=51, sigma=0.5):
    return nn.Sequential(
        nn.Scale(1 / 255),
        nn.Conv2d(frames, 32, 8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, 4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.CategoricalDueling(
            nn.Sequential(
                nn.NoisyFactorizedLinear(3136, hidden, sigma_init=sigma),
                nn.ReLU(),
                nn.NoisyFactorizedLinear(
                    hidden,
                    atoms,
                    init_scale=0,
                    sigma_init=sigma
                )
            ),
            nn.Sequential(
                nn.NoisyFactorizedLinear(3136, hidden, sigma_init=sigma),
                nn.ReLU(),
                nn.NoisyFactorizedLinear(
                    hidden,
                    env.action_space.n * atoms,
                    init_scale=0,
                    sigma_init=sigma
                )
            )
        )
    )


class impala_cnn_features(nn.Module):
    def __init__(self, env, channels=1):
        super().__init__()
        conv_layers = nn.ModuleList()
        prev_num_ch = channels
        for i, (num_ch, num_blocks) in enumerate([(16, 2), (32, 2), (64, 2), (64, 2)]):
            cur_layer = nn.ModuleDict()
            cur_layer["o"] = nn.Conv2d(prev_num_ch, num_ch, 3, stride=1, padding=1)
            cur_layer["p"] = nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
            blocks = nn.ModuleList()
            for b in range(num_blocks):
                block = nn.ModuleList([
                    nn.Conv2d(num_ch, num_ch, 3, stride=1, padding=1),
                    nn.Conv2d(num_ch, num_ch, 3, stride=1, padding=1),
                ])
                blocks.append(block)

            cur_layer["b"] = blocks
            prev_num_ch = num_ch
            conv_layers.append(cur_layer)

        self.conv_layers = conv_layers

    def forward(self, input):
        conv_layers = self.conv_layers
        cur_conv = input.float()/255.
        for i in range(4):
            cur_layer = conv_layers[i]
            cur_conv = cur_layer["o"](cur_conv)
            cur_conv = cur_layer["p"](cur_conv)
            for block in cur_layer["b"]:
                block_input = cur_conv
                for layer in block:
                    cur_conv = torch.relu(cur_conv)
                    cur_conv = layer(cur_conv)
                cur_conv += block_input
        cur_conv = cur_conv.reshape(cur_conv.shape[0], -1)
        return cur_conv

class impala_rnn_features(nn.Module):
    def __init__(self, env, length=4, channels=1):
        super().__init__()
        self.length = length
        self.channels = channels
        self.out_size = 3136
        self.lstm_size = 256
        self.cnn_encoder = nn.Sequential(
            nn.Scale(1 / 255),
            nn.Conv2d(channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.Flatten(),
        )
        # impala_cnn_features(env, channels)
        self.initial_cell = nn.parameter.Parameter(torch.zeros((1,1,self.lstm_size)), requires_grad=True)
        self.initial_hidden = nn.parameter.Parameter(torch.zeros((1,1,self.lstm_size)), requires_grad=True)
        self.rnn = nn.LSTM(self.out_size, self.lstm_size, 1)

    def to(self, device):
        super().to(device)

    def forward(self, input):
        batch_size = input.shape[0]
        cnn_input = input.reshape(batch_size * self.length, self.channels, *input.shape[2:])
        cnn_output = self.cnn_encoder(cnn_input)
        cnn_output = cnn_output.reshape(batch_size, self.length, self.out_size)
        cnn_output = cnn_output.permute((1, 0, 2))
        rnn_outs, (hn, cn) = self.rnn(cnn_output, (torch.tile(self.initial_hidden, (1,batch_size,1)), torch.tile(self.initial_cell, (1,batch_size,1))))
        final_out = rnn_outs[-1]
        return final_out



# net = impala_rnn_features(None, 7, 1)
# # torch.save(net,"arg.pt")
# net = net.to("cuda")
# out = net(torch.ones((2,7,84,84),device="cuda"))

# print(out.shape)
def impala_features(length=4, channels=4):
    return nn.Sequential(
        impala_rnn_features(None, length=length, channels=channels),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU()
    )


def impala_value_head():
    return nn.Linear(256, 1)


def impala_policy_head(env):
    return nn.Linear0(256, env.action_space.n)


def impala_rainbow(env,frames=None, length=4, channels=4, hidden=512, atoms=51, sigma=0.5):
    return nn.Sequential(
        impala_rnn_features(env, length=length, channels=channels),
        nn.CategoricalDueling(
            nn.Sequential(
                nn.NoisyFactorizedLinear(256, hidden, sigma_init=sigma),
                nn.ReLU(),
                nn.NoisyFactorizedLinear(
                    hidden,
                    atoms,
                    init_scale=0,
                    sigma_init=sigma
                )
            ),
            nn.Sequential(
                nn.NoisyFactorizedLinear(256, hidden, sigma_init=sigma),
                nn.ReLU(),
                nn.NoisyFactorizedLinear(
                    hidden,
                    env.action_space.n * atoms,
                    init_scale=0,
                    sigma_init=sigma
                )
            )
        )
    )

class Dueling(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, value_branch, advantage_branch):
        super().__init__()
        self.flatten = nn.Flatten()
        self.value_branch = value_branch
        self.advantage_branch = advantage_branch

    def forward(self, x, advantages_only=False):
        x = self.flatten(x)
        advantages = self.advantage_branch(x)
        if advantages_only:
            return advantages

        value = self.value_branch(x)
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))


class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth, norm_func):
        super().__init__()

        self.relu = nn.ReLU()
        self.conv_0 = (nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        x_ = self.conv_0(self.relu(x))
        x_ = self.conv_1(self.relu(x_))
        return x+x_

def identity(p):
    return p

class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func):
        super().__init__()

        self.conv = norm_func(nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1))
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)
        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=identity)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func)

    def forward(self, x):
        x = self.conv(x)
        x = self.max_pool(x)
        x = self.residual_0(x)
        x = self.residual_1(x)
        return x


class ImpalaCNNLarge(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, linear_layer, resolution, model_size=1, spectral_norm=False):
        super().__init__()

        norm_func = torch.nn.utils.spectral_norm if spectral_norm else identity

        self.main = nn.Sequential(
            ImpalaCNNBlock(in_depth, 16*model_size, norm_func=identity),
            ImpalaCNNBlock(16*model_size, 32*model_size, norm_func=identity),
            ImpalaCNNBlock(32*model_size, 32*model_size, norm_func=norm_func),
            nn.ReLU(),
            nn.Flatten(),
        )

        shape = self.main(torch.zeros(1, in_depth, resolution[0], resolution[1])).shape
        assert shape[0] == 1
        self.linear = linear_layer(shape[1], 256)
        # print(shape[1])
        # print(model_size)
        # assert shape[1] == 32*np.prod(model_size)
        #
        # self.dueling = Dueling(
        #     nn.Sequential(linear_layer(shape[2]*shape[3]*32*model_size, 256),
        #                   nn.ReLU(),
        #                   linear_layer(256, 1)),
        #     nn.Sequential(linear_layer(shape[2]*shape[3]*32*model_size, 256),
        #                   nn.ReLU(),
        #                   linear_layer(256, actions))
        # )

    def forward(self, x, advantages_only=False):
        f = self.main(x)
        l = self.linear(f)
        # print(f.shape)
        return torch.relu(l)
        # return self.dueling(f, advantages_only=advantages_only)


if __name__ == "__main__":
    # test out net to make sure it at least can be created
    largenet = ImpalaCNNLarge(4, 18, nn.Linear, (84, 84), model_size=2)
    out = largenet(torch.zeros(2,4,84,84))
    print(out)
    print(out.shape)
