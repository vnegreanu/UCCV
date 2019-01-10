import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        #set the hidden dimension
        self.hidden_dim = hidden_size
        
        #set the internal embeddings as a lookup table from the number of words to train and size of the vectors
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        
        #set the LSTM
        self.lstm = nn.LSTM(input_size = embed_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)
        
        #set the output layer
        self.linear = nn.Linear(in_features = hidden_size, 
                                out_features = vocab_size)
    
    def forward(self, features, captions):
         
        #trim the last column in captions    
        captions = captions[:, :-1]    
            
        #tensor of last batches    
        embed = self.embeddings(captions)
        
        #concat the features and captions
        embed = torch.cat((features.unsqueeze(dim = 1), embed), dim=1)
        
        #get the output and hidden state by passing over lstm
        lstm_out , self.hidden = self.lstm(embed)
        
        #last the fully connected layer
        output = self.linear(lstm_out)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        
        for index in range(max_len):
            
            #run through the lstm 
            lstm_out, states = self.lstm(inputs, states) # (1, 1, states_size)
            lstm_out = lstm_out.squeeze(1)               # (1, 1, vocabulary_size)    
            output = self.linear(lstm_out)               # (1, vocabulary_size) 
            
            #append to sentence according to max probabilities
            sentence.append(output.max(1)[1].item())
            
            #update inputs
            inputs = self.embeddings(output.max(1)[1]) # (1, embed_size)
            inputs = inputs.unsqueeze(1)               # (1, 1, embed_size)
            
        return sentence