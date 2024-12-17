import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from module.addbatch import addbatch

def train_test(train_input, train_label, val_input, val_label, test_input, test_label, net, optimizer, loss_func, batch_size, model_path='best_model.pth'):
    traindata = addbatch(train_input, train_label, batch_size)
    best_val_accuracy = 0.0
    train_losses = []
    val_losses = []

    if torch.cuda.is_available():
        net.cuda()
        train_input, train_label = train_input.cuda(), train_label.cuda()
        val_input, val_label = val_input.cuda(), val_label.cuda()
        test_input, test_label = test_input.cuda(), test_label.cuda()

    for epoch in range(301):
        net.train()
        running_loss = 0.0
        for step, data in enumerate(traindata):
            inputs, labels = data
            # print('labels',labels)
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            out = net(inputs)
            # print('out:',out)
            train_loss = loss_func(out, labels)
            train_loss.backward()
            optimizer.step()
            running_loss += train_loss.item()

        epoch_train_loss = running_loss / len(traindata)
        train_losses.append(epoch_train_loss)

        net.eval()
        with torch.no_grad():
            val_out = net(val_input)
            val_loss = loss_func(val_out, val_label)
            val_losses.append(val_loss.item())
            val_prediction = torch.max(val_out, 1)[1].cpu().numpy()
            val_accuracy = accuracy_score(val_label.cpu().numpy(), val_prediction)

        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f'Epoch [{epoch + 1}/3001], Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss.item():.4f}, Val Accuracy: {val_accuracy:.4f}')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(net.state_dict(), model_path)
            print(f'Saved Best Model at Epoch {epoch + 1} with Val Accuracy: {val_accuracy:.4f}')

    net.load_state_dict(torch.load(model_path))
    print('Loaded Best Model')

    test_out = net(test_input)
    test_prediction = torch.max(test_out, 1)[1].cpu().numpy()
    test_label_np = test_label.cpu().numpy()

    cm_test = confusion_matrix(test_label_np, test_prediction)

    test_probs = F.softmax(test_out, dim=1).cpu().detach().numpy()
    y_test_pred = test_prediction.tolist()

    val_probs = F.softmax(val_out, dim=1).cpu().detach().numpy()
    y_val_pred = val_prediction.tolist()

    cm_val = confusion_matrix(val_label.cpu().numpy(), val_prediction)

    return cm_val, cm_test, val_probs, test_probs, y_val_pred, y_test_pred, train_losses, val_losses


