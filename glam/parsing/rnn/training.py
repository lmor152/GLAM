# from torch.utils.tensorboard import SummaryWriter
# import torch
# import torch.nn as nn

# from glam.parsing.rnn.model import address_parser_model

# writer = SummaryWriter(comment='Model E long')
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = nn.CrossEntropyLoss(ignore_index = -1)

# Training the model

# address_parser_model.train()
# for i, batch in enumerate(train_dataloader):
#     input_data, target_data = batch
#     input_data, target_data = input_data.to(device), target_data.to(device)
    
#     optimizer.zero_grad()
#     output = model(input_data)
    
#     # Reshape target_data and output for Loss Function
#     output = output.view(-1, output.shape[ -1 ])  # Shape : (batch_size * seq_len, num_classes)
#     target_data = target_data.view(-1)  # Shape : (batch_size * seq_len)
    
#     loss = criterion(output, target_data)
#     writer.add_scalar('Loss/train', loss, i)
    
#     loss.backward()
#     optimizer.step()