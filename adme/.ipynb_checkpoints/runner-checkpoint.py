def run(args):
    from src.tdc_constant import TDC
    from dotmap import DotMap
    def set_model_name():
        from datetime import datetime
        li = []
        for tdc in TDC.allList:
            li.append(str(tdc) if tdc in args.data else 'X')
        name = ', '.join(li)
        return f'MolCLR_[{name}]{"_sc" if args.scaled else ""}-{datetime.now().strftime("%m.%d_%H%M")}'


    model_name= args.load_config.name if args.load_config.load else set_model_name()
    modelf=f'ckpts/{model_name}.pt'

    model_name, modelf

    args.data_config = DotMap({
        'batch_size':256, # Don't know well
        'num_workers':0,
    })

    from src.dataset_mtl import MolTestDatasetWrapper
    h_dataset = MolTestDatasetWrapper(
        tdcList = args.data,
        scaled = args.scaled,
        batch_size = args.data_config.batch_size,
        num_workers = args.data_config.num_workers,
    )

    # trainloader: torch_geometric.loader.DataLoader
    trainloader,validloader,testloader=h_dataset.get_data_loaders()

    # len = row / batch_size
    len(trainloader), len(validloader), len(testloader)

    from src.ginet_finetune import GINet_Feat_MTL, load_pre_trained_weights

    model = GINet_Feat_MTL(
        pool = 'mean',
        drop_ratio = args.learning_config.gin_drop_ratio,
        pred_layer_depth = args.learning_config.pred_layer_depth,
        num_tasks = args.learning_config.num_tasks,
        pred_act = 'relu',
    ).to(args.device)
    model = load_pre_trained_weights(model, args.device, '../aigenintern1/23-2/MolCLR/pretrained_weights/pretrained_gin_model.pth')

    # set different learning rates for prediction head and base

    # 1) check if model_parameters are learnable
    layer_list = [] # layer_list = prediction head
    for name, param in model.named_parameters():
        if 'pred_head' in name:
            print(name, param.requires_grad)
            layer_list.append(name)

    # 2) set different learning rates for prediction head and base
    # params: prediction head
    params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
    # base_params: base
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

    import torch
    optimizer = torch.optim.Adam(
        [
            {'params': base_params, 'lr': args.learning_config.init_base_lr},
            {'params': params}
        ],
        args.learning_config.init_pred_lr,
        weight_decay = args.learning_config.weight_decay
    )

    from torch import nn

    criterions = []
    for tdc in args.data:
        if tdc.isRegression():
            criterions.append(nn.MSELoss())
        else:
            criterions.append(nn.BCEWithLogitsLoss())

    import math
    def mtl_loss(pred, label, criterions):
        li = []
        # loss_i = criterion(pred[:,i].squeeze(), label[:,i].squeeze())
        for i in range(args.learning_config.num_tasks):
            label_task = label[:, i].squeeze()
            pred_task = pred[:, i].squeeze()
            mask = ~torch.isnan(label_task)
            # Loss is already divided by len(pred_task[mask]) by nn.MSELoss or nn.BCELoss
            # This resolves the problem of potential bias in the loss due to different
            # numbers of labels across tasks. The normalization ensures that the loss is
            # scale-invariant with respect to the number of elements, making it fair and
            # comparable across tasks, even when they have different numbers of labels.
            x = criterions[i](pred_task[mask], label_task[mask])

            if not math.isnan(x): # nan if len(pred_task[mask]) == 0
                li.append(x)
            else:
                li.append(torch.tensor(0, device=args.device, dtype=torch.float32))

        # loss = mean of each mtl loss & batch
        loss = torch.mean(torch.stack(li), dim=0)
        return loss

    def train(model, trainloader, args, optimizer=optimizer, criterions=criterions):
        model.train() # set to train mode
        train_loss = 0
        for batch in trainloader:
            batch = batch.to(args.device)
            label = batch.y

            optimizer.zero_grad()
            pred = model(batch)

            loss = mtl_loss(pred, label, criterions)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(trainloader)
        return avg_train_loss

    def eval(model, loader, args, criterions=criterions):
        model.eval()  # Set to eval mode
        eval_loss = 0
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(args.device)
                label = batch.y
                pred = model(batch)

                eval_loss += mtl_loss(pred, label, criterions).item()
        avg_eval_loss = eval_loss / len(loader)

        return avg_eval_loss

    from datetime import datetime
    if args.load_config.load == False:
        epoch = 0
        print_every_n_epoch = 5

        from src.EarlyStopper import EarlyStopper
        early_stopper = EarlyStopper(patience=args.learning_config.patience,printfunc=print, 
                                     verbose=False, path=modelf)

        with open(f'Log: {model_name}.txt', 'a') as fp:
            fp.write('Start Training\n')
            fp.write(f'{model_name}\n')
            fp.write(f'{args.device}\n')
            fp.flush()
            while True:
                epoch+=1
                train_loss=train(model,trainloader,args)
                valid_loss=eval(model,validloader,args)
                fp.write(f'[Epoch{epoch}] train_loss: {train_loss:.4f}, valid_loss: {valid_loss:.4f}. {datetime.now().strftime("%H:%M:%S")}\n')
                fp.flush()
                if (epoch % print_every_n_epoch == 0):
                    pass

                early_stopper(valid_loss,model)
                if early_stopper.early_stop:
                    fp.write('early stopping\n')
                    fp.flush()
                    break
    else:
        print('Skip Training')