from torch.nn import Module

NNprint_= False

class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class UnFlatten_FF(Module):
    def forward(self, input, size=784):
        return input.view(-1,1,28,28)

# Debugging module
class NNprint(Module):
    def forward(self, input):
        if NNprint_==True:
            print(input.shape)
        return input
