from torch import optim
from yolo.dataset import *
from torch.utils.data import DataLoader
from yolo.yolo_v3_net import *
from torch.utils.tensorboard import SummaryWriter

def loss_fun(output,target):
    output = output.permute(0,2,3,1)
    output = output.reshape(output.size(0),output.size(1),output.size(2),3,-1)

    mask_obj = target[...,0] > 0
    mask_no_obj = target[...,0] == 0
    loss_p_fun = nn.BCELoss()
    loss_obj = loss_p_fun(torch.sigmoid(output[mask_obj][...,0]),target[mask_obj][...,0])
    loss_noobj= loss_p_fun(torch.sigmoid(output[mask_no_obj][...,0]),target[mask_no_obj][...,0])

    loss_box_fun = nn.MSELoss()
    loss_box = loss_box_fun(output[mask_obj][...,1:5],target[mask_obj][...,1:5])

    loss_cls_fun = nn.CrossEntropyLoss()
    loss_cls = loss_cls_fun(output[mask_obj][...,5:],torch.argmax(target[mask_obj][...,5:],dim=1,keepdim=True).squeeze(dim=1))

    loss = loss_obj + 100*loss_noobj + loss_box + loss_cls
    return loss


if __name__ == '__main__':
    summary_writer = SummaryWriter('logs')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = YoloDataSet()
    data_loader = DataLoader(dataset,batch_size=2,shuffle=True)

    weight_path = 'params/net.pt'
    net = Yolo_V3_Net().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))

    opt = optim.Adam(net.parameters())

    epoch = 0
    max_epoch = 100
    while True:
        if epoch == max_epoch:
            break
        index = 0
        for tgt_13,tgt_26,tgt_52,img_data in data_loader:
            tgt_13,tgt_26,tgt_52,img_data = tgt_13.to(device),tgt_26.to(device),tgt_52.to(device),img_data.to(device)
            output_13,output_26,output_52 = net(img_data)

            loss_13 = loss_fun(output_13.float(),tgt_13.float())
            loss_26 = loss_fun(output_26.float(),tgt_26.float())
            loss_52 = loss_fun(output_52.float(),tgt_52.float())
            loss = loss_13 + loss_26 + loss_52

            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f'loss{epoch}=={index}', loss.item())
            summary_writer.add_scalar('train_loss', loss, index)
            index += 1

        torch.save(net.state_dict(), 'params/net.pt')
        print('模型保存成功')
        epoch += 1
