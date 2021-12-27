import argparse
import time

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

import test  # import test.py to get mAP after each epoch
from models import *
from datasets import *
from utils import *

#      0.149      0.241      0.126      0.156       6.85      1.008      1.421    0.07989      16.94      6.215      10.61      4.272      0.251      0.001         -4        0.9     0.0005   320 64-1 giou
hyp = {'giou': 1.008,  # giou loss gain
       'xy': 1.421,  # xy loss gain
       'wh': 0.07989,  # wh loss gain
       'cls': 16.94,  # cls loss gain
       'cls_pw': 6.215,  # cls BCELoss positive_weight
       'conf': 10.61,  # conf loss gain
       'conf_pw': 4.272,  # conf BCELoss positive_weight
       'iou_t': 0.251,  # iou target-anchor training threshold
       'lr0': 0.001,  # initial learning rate
       'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
       'momentum': 0.90,  # SGD momentum
       'weight_decay': 0.0005}  # optimizer weight decay


#     0.0945      0.279      0.114      0.131         25      0.035        0.2        0.1      0.035         79       1.61       3.53       0.29      0.001         -4        0.9     0.0005   320 64-1
#     0.112       0.265      0.111      0.144       12.6      0.035        0.2        0.1      0.035         79       1.61       3.53       0.29      0.001         -4        0.9     0.0005   320 32-2
# hyp = {'giou': .035,  # giou loss gain
#        'xy': 0.20,  # xy loss gain
#        'wh': 0.10,  # wh loss gain
#        'cls': 0.035,  # cls loss gain
#        'cls_pw': 79.0,  # cls BCELoss positive_weight
#        'conf': 1.61,  # conf loss gain
#        'conf_pw': 3.53,  # conf BCELoss positive_weight
#        'iou_t': 0.29,  # iou target-anchor training threshold
#        'lr0': 0.001,  # initial learning rate
#        'lrf': -4.,  # final learning rate = lr0 * (10 ** lrf)
#        'momentum': 0.90,  # SGD momentum
#        'weight_decay': 0.0005}  # optimizer weight decay

def writeLogToFile(text, outputPath, taskName):

    print(text)
    #------------------------------------
    # get current timestamp
    #------------------------------------
    ts = time.gmtime()
    currentDateTime = time.strftime("%Y%m%d_%H%M%S", ts)

    #------------------------------------
    # text file
    #------------------------------------
    logFilePath = outputPath + 'log_' + taskName + '_' + currentDateTime + '.txt'
    with open(logFilePath, 'a') as fileHandler:
        fileHandler.write(text + '\n')

def writePredToFile(pred, outputFilePath):
    with open(outputFilePath, 'w') as fileHandler:
        for item in pred:
            fileHandler.write(str(item) + '\n')

def readPredFromFile(filePath):
    pred = []
    with open(filePath, 'r') as fileHandler:
        for line in fileHandler:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]
            tensorItem = [float(currentPlace)]
            tensorItem = np.asarray(tensorItem)
            tensorItem = torch.from_numpy(tensorItem)

            # add item to the list
            pred.append(tensorItem)

    return pred

def generate_prediction_of_old_model(oldCfg, imgSize, device, oldWeightPath,imgs):
    # Initialize model
    model = Darknet(oldCfg, imgSize).to(device)

    # Load weights
    if oldWeightPath.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(oldWeightPath, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, oldWeightPath)

    model.eval()
    inf_out, train_out = model(imgs)


    return inf_out, train_out

def train(
        cfg,
        data_cfg,
        output_path,
        old_cfg,
        old_weight_path,
        img_size=416,
        epochs=100,  # 500200 batches at bs 16, 117263 images = 273 epochs
        batch_size=8,
        accumulate=8,  # effective bs = batch_size * accumulate = 8 * 8 = 64
        freeze_backbone=False,
):
    init_seeds()
    weights = output_path + 'weights' + os.sep
    latest = weights + 'latest.pt'
    best = weights + 'best.pt'
    device = torch_utils_lwf.select_device()
    img_size_test = img_size  # image size for testing
    multi_scale = not opt.single_scale

    if multi_scale:
        img_size_min = round(img_size / 32 / 1.5)
        img_size_max = round(img_size / 32 * 1.5)
        img_size = img_size_max * 32  # initiate with maximum multi_scale size

    # Configure run
    data_dict = parse_data_cfg(data_cfg)
    train_path = data_dict['train']
    nc = int(data_dict['classes'])  # number of classes

    # Initialize model
    model = Darknet(cfg).to(device)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=hyp['lr0'], momentum=hyp['momentum'], weight_decay=hyp['weight_decay'])

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = 0.0
    nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)
    chkpt = torch.load(old_weight_path, map_location=device)
    model.load_state_dict({k: v for k, v in chkpt['model'].items() if v.numel() > 1 and v.shape[0] != 255},
                          strict=False)
    for p in model.parameters():
        p.requires_grad = True if p.shape[0] == nf else False

    start_epoch = chkpt['epoch'] + 1
    if chkpt['optimizer'] is not None:
        optimizer.load_state_dict(chkpt['optimizer'])
        best_fitness = chkpt['best_fitness']
    del chkpt

    # Scheduler https://github.com/ultralytics/yolov3/issues/238
    # lf = lambda x: 1 - x / epochs  # linear ramp to zero
    # lf = lambda x: 10 ** (hyp['lrf'] * x / epochs)  # exp ramp
    # lf = lambda x: 1 - 10 ** (hyp['lrf'] * (1 - x / epochs))  # inverse exp ramp
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[round(opt.epochs * x) for x in (0.8, 0.9)], gamma=0.1)
    scheduler.last_epoch = start_epoch - 1

    # # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Dataset
    rectangular_training = False
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  rect=rectangular_training)

    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank

        model = torch.nn.parallel.DistributedDataParallel(model)
        # sampler = torch.utils.data.distributed.DistributedSampler(dataset)

    # Dataloader
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=opt.num_workers,
                            shuffle=not rectangular_training,  # Shuffle=True unless rectangular training is used
                            pin_memory=True,
                            collate_fn=dataset.collate_fn)

    # Mixed precision training https://github.com/NVIDIA/apex
    mixed_precision = True
    if mixed_precision:
        try:
            from apex import amp
            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        except:  # not installed: install help: https://github.com/NVIDIA/apex/issues/259
            mixed_precision = False

    # Start training
    model.hyp = hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    model_info(model, report='summary')  # 'full' or 'summary'
    nb = len(dataloader)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0)  # P, R, mAP, F1, test_loss
    n_burnin = min(round(nb / 5 + 1), 1000)  # burn-in batches
    t, t0 = time.time(), time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        print(('\n%8s%12s' + '%10s' * 7) %
              ('Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'targets', 'img_size'))

        # Update scheduler
        scheduler.step()

        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        # # Update image weights (optional)
        # w = model.class_weights.cpu().numpy() * (1 - maps)  # class weights
        # image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
        # dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # random weighted index

        mloss = torch.zeros(5).to(device)  # mean losses
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        for i, (imgs, targets, _, _) in pbar:
            imgs = imgs.to(device)
            targets = targets.to(device)
            print('\n')
            print('==================================')
            print('GROUND TRUTH')
            print('==================================')
            print('size: ' + str(len(targets)) + ' x ' + str(len(targets[0])))
            # print('Ground truth: ' +str(targets))

            # Multi-Scale training
            if multi_scale:
                if (i + nb * epoch) / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.choice(range(img_size_min, img_size_max + 1)) * 32
                    # print('img_size = %g' % img_size)
                scale_factor = img_size / max(imgs.shape[-2:])
                imgs = F.interpolate(imgs, scale_factor=scale_factor, mode='bilinear', align_corners=False)

            # Plot images with bounding boxes
            if epoch == 0 and i == 0:
                plot_images(imgs=imgs, targets=targets, fname='train_batch%g.jpg' % i)

            # SGD burn-in
            if epoch == 0 and i <= n_burnin:
                lr = hyp['lr0'] * (i / n_burnin) ** 4
                for x in optimizer.param_groups:
                    x['lr'] = lr

            # generate prediction of old model
            print('Generate prediction of old model')
            old_inf_out, old_train_out = generate_prediction_of_old_model(old_cfg, img_size, device, old_weight_path,imgs)
            # print('inf_out: ' + str(old_inf_out))
            # print('train_out: ' + str(old_train_out))
            print('==================================')
            print('OLD PREDICTION')
            print('==================================')
            print('size of inference: ' + str(len(old_inf_out)) + ' x ' + str(len(old_inf_out[0])) + ' x ' + str(len(old_inf_out[0][0])))
            print('size of training: ' + str(len(old_train_out)) + ' x ' + str(len(old_train_out[0])) + ' x ' + str(len(old_train_out[0][0])) + ' x ' + str(len(old_train_out[0][0][0])) + ' x ' + str(len(old_train_out[0][0][0][0])) + ' x ' + str(len(old_train_out[0][0][0][0][0])))

            # Run model
            print('Run model')
            new_pred = model(imgs)
            print('==================================')
            print('CURRENT PREDICTION')
            print('==================================')
            print('size: ' + str(len(new_pred)) + ' x ' + str(len(new_pred[0])) + ' x ' + str(len(new_pred[0][0])) + ' x ' + str(len(new_pred[0][0][0])) + ' x ' + str(len(new_pred[0][0][0][0])) + ' x ' + str(len(new_pred[0][0][0][0][0])))
            # print('Prediction: ' + str(new_pred))
            # print('Prediction data type: ' + str(type(pred)))
            # predFileName = 'pred_' + str(epoch) +'.txt'

            # writePredToFile(pred, outputPath+predFileName)

            # # read prediction from file again -- TEST
            # pred = readPredFromFile(outputPath+predFileName)

            # Compute distillation loss
            print('Compute distillation loss')
            loss, loss_items = compute_distillation_loss(old_inf_out, new_pred, targets, model, giou_loss=opt.giou)

            # loss, loss_items = compute_loss(pred, targets, model, giou_loss=opt.giou)
            if torch.isnan(loss):
                print('WARNING: nan loss detected, ending training')
                return results

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == nb:
                optimizer.step()
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            # s = ('%8s%12s' + '%10.3g' * 7) % ('%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nb - 1), *mloss, len(targets), time.time() - t)
            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1), '%g/%g' % (i, nb - 1), *mloss, len(targets), img_size)
            t = time.time()
            pbar.set_description(s)  # print(s)

        # Report time
        dt = (time.time() - t0) / 3600
        print('%g epochs completed in %.3f hours.' % (epoch - start_epoch + 1, dt))

        # Calculate mAP (always test final epoch, skip first 5 if opt.nosave)
        if not (opt.notest or (opt.nosave and epoch < 10)) or epoch == epochs - 1:
            with torch.no_grad():
                results, maps = test.test(cfg, data_cfg, batch_size=batch_size, img_size=img_size_test, model=model,
                                          conf_thres=0.1)

        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 5 % results + '\n')  # P, R, mAP, F1, test_loss

        # Update best map
        fitness = results[2]
        if fitness > best_fitness:
            best_fitness = fitness

        # Save training results
        save = (not opt.nosave) or (epoch == epochs - 1)
        if save:
            # Create checkpoint
            chkpt = {'epoch': epoch,
                     'best_fitness': best_fitness,
                     'model': model.module.state_dict() if type(
                         model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                     'optimizer': optimizer.state_dict()}

            # Save latest checkpoint
            torch.save(chkpt, latest)

            # Save best checkpoint
            if best_fitness == fitness:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            if epoch > 0 and epoch % 10 == 0:
                torch.save(chkpt, weights + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

    return results


def print_mutation(hyp, results):
    # Write mutation results
    a = '%11s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%11.4g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    c = '%11.3g' * len(results) % results  # results (P, R, mAP, F1, test_loss)
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    if opt.cloud_evolve:
        os.system('gsutil cp gs://yolov4/evolve.txt .')  # download evolve.txt
        with open('evolve.txt', 'a') as f:  # append result to evolve.txt
            f.write(c + b + '\n')
        os.system('gsutil cp evolve.txt gs://yolov4')  # upload evolve.txt
    else:
        with open('evolve.txt', 'a') as f:
            f.write(c + b + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', type=str, default='test_lwf', help='Task name')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--accumulate', type=int, default=4, help='number of batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='E:/research/_mine/yolov3-pytorch/dataset/coco/darknet/21_add_backpack/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data_cfg', type=str, default='E:/research/_mine/yolov3-pytorch/dataset/coco/darknet/21_add_backpack/local/base_voc_local.data', help='base_voc.data file path')
    parser.add_argument('--output_path', type=str, default='E:/research/_mine/yolov3-pytorch/weights/local_test/lwf/', help='output file path')
    parser.add_argument('--old_cfg', type=str, default='E:/research/_mine/yolov3-pytorch/dataset/baseline_voc/yolov3.cfg', help='old cfg file path')
    parser.add_argument('--old_weight_path', type=str, default='E:/research/_mine/yolov3-pytorch/dataset/baseline_voc/weights/best-cpu.pt', help='old weight file path')
    parser.add_argument('--single_scale', action='store_true', help='train at fixed size (no multi-scale)')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--num_workers', type=int, default=4, help='number of Pytorch DataLoader workers')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--giou', action='store_true', help='use GIoU loss instead of xy, wh loss')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cloud_evolve', action='store_true', help='evolve hyperparameters from a cloud source')
    parser.add_argument('--var', default=0, type=int, help='debug variable')
    opt = parser.parse_args()

    #write training log into a text file
    writeLogToFile('== CONTINUOUS LEARNING ==', opt.output_path, opt.task_name)
    writeLogToFile('== Adding one new class ==', opt.output_path, opt.task_name)

    writeLogToFile('Task name: ' + opt.task_name, opt.output_path, opt.task_name)
    writeLogToFile('CURRENT MODEL', opt.output_path, opt.task_name)
    writeLogToFile('Model configuration: ' + opt.cfg, opt.output_path, opt.task_name)
    writeLogToFile('Training data: ' + opt.data_cfg, opt.output_path, opt.task_name)
    writeLogToFile('Output path: ' + opt.output_path, opt.output_path, opt.task_name)
    writeLogToFile('/n', opt.output_path, opt.task_name)
    writeLogToFile('PREVIOUS MODEL', opt.output_path, opt.task_name)
    writeLogToFile('Previous model configuration: ' + opt.old_cfg, opt.output_path, opt.task_name)
    writeLogToFile('Previous model data: ' + opt.old_weight_path, opt.output_path, opt.task_name)
    writeLogToFile('/n', opt.output_path, opt.task_name)
    writeLogToFile('======================================================', opt.output_path, opt.task_name)
    writeLogToFile('Training parameters:', opt.output_path, opt.task_name)
    writeLogToFile('======================================================', opt.output_path, opt.task_name)
    writeLogToFile('Epochs = ' + str(opt.epochs), opt.output_path, opt.task_name)
    writeLogToFile('Batch size = ' + str(opt.batch_size), opt.output_path, opt.task_name)
    writeLogToFile('Number of batches to accumulate = ' + str(opt.accumulate), opt.output_path, opt.task_name)
    if(opt.single_scale):
        writeLogToFile('Fixed-size training', opt.output_path, opt.task_name)
    else:
        writeLogToFile('Multi-scale training', opt.output_path, opt.task_name)
    writeLogToFile('Inference size (pixels) = ' + str(opt.img_size), opt.output_path, opt.task_name)
    writeLogToFile('Number of Pytorch DataLoader workers' + str(opt.num_workers), opt.output_path, opt.task_name)

    # Train
    results = train(opt.cfg,
                    opt.data_cfg,
                    opt.output_path,
                    opt.old_cfg,
                    opt.old_weight_path,
                    img_size=opt.img_size,
                    epochs=opt.epochs,
                    batch_size=opt.batch_size,
                    accumulate=opt.accumulate)

    # Evolve hyperparameters (optional)
    if opt.evolve:
        gen = 1000  # generations to evolve
        print_mutation(hyp, results)  # Write mutation results

        for _ in range(gen):
            # Get best hyperparamters
            x = np.loadtxt('evolve.txt', ndmin=2)
            x = x[x[:, 2].argmax()]  # select best mAP as genetic fitness (col 2)
            for i, k in enumerate(hyp.keys()):
                hyp[k] = x[i + 5]

            # Mutate
            init_seeds(seed=int(time.time()))
            s = [.2, .2, .2, .2, .2, .2, .2, .2, .2 * 0, .2 * 0, .05 * 0, .2 * 0]  # fractional sigmas
            for i, k in enumerate(hyp.keys()):
                x = (np.random.randn(1) * s[i] + 1) ** 2.0  # plt.hist(x.ravel(), 300)
                hyp[k] *= float(x)  # vary by 20% 1sigma

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay']
            limits = [(1e-4, 1e-2), (0, 0.70), (0.70, 0.98), (0, 0.01)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train(opt.cfg,
                            opt.data_cfg,
                            img_size=opt.img_size,
                            epochs=opt.epochs,
                            batch_size=opt.batch_size,
                            accumulate=opt.accumulate)

            # Write mutation results
            print_mutation(hyp, results)

            # # Plot results
            # import numpy as np
            # import matplotlib.pyplot as plt
            # a = np.loadtxt('evolve_1000val.txt')
            # x = a[:, 2] * a[:, 3]  # metric = mAP * F1
            # weights = (x - x.min()) ** 2
            # fig = plt.figure(figsize=(14, 7))
            # for i in range(len(hyp)):
            #     y = a[:, i + 5]
            #     mu = (y * weights).sum() / weights.sum()
            #     plt.subplot(2, 5, i+1)
            #     plt.plot(x.max(), mu, 'o')
            #     plt.plot(x, y, '.')
            #     print(list(hyp.keys())[i],'%.4g' % mu)
