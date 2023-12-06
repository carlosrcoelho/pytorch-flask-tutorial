import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io
from PIL import Image

# load model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out
    
    
# Hyper-parameters 
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10    

model = NeuralNet(input_size, hidden_size, num_classes)

PATH = 'app/mnist_ffn.pth'
model.load_state_dict(torch.load(PATH))
model.eval()


# image -> tensor

def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),  # means one channel
                                    transforms.Resize((28,28)),  # resize the image to 28x28 pixels
                                    transforms.ToTensor(),  # transform it into a PyTorch tensor
                                    transforms.Normalize((0.1307,), (0.3081,))])  # normalize the image
    image = Image.open(io.BytesIO(image_bytes))  # convert bytes data into PIL image
    return transform(image).unsqueeze(0)   # unsqueeze to add artificial first dimension for batch

# predict

def get_prediction(image_tensor):
    images = image_tensor.reshape(-1, 28*28)
    outputs = model(images)
        # max returns (value ,index)
    _, predicted = torch.max(outputs.data, 1)
    return predicted