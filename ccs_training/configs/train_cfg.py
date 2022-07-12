cfg = {
        # data
        'train_file': 'bmyan/dataset/CelebAMask-HQ/train_annos/train_attrbute_9.txt',
        'valid_file': 'bmyan/dataset/CelebAMask-HQ/train_annos/valid_attrbute_9.txt',
        'da_anno_file_list' : [
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM1L4.json',      #同类，单层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM1L8.json',      #同类，单层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM1L12.json',     #同类，单层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM2Ltop4.json',   #同类，多层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM2Ltop8.json',   #同类，多层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM2Ltop12.json',  #同类，多层交换

            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM1L4.json',      #同类，单层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM1L6.json',      #同类，单层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM1L8.json',      #同类，单层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM1L10.json',     #同类，单层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM1L12.json',     #同类，单层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM2Ltop4.json',   #同类，多层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM2Ltop6.json',   #同类，多层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM2Ltop8.json',   #同类，多层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM2Ltop10.json',  #同类，多层交换
            # 'leogb/causal/CelebAMask-HQ-ComAttr3-19-32/editingM2Ltop12.json',  #同类，多层交换
        ],
        'input_size': (224, 224),
        'num_workers': 8,
        'val_num_workers': 8,
        'test_num_workers': 8,
        'batch_size': 32,
        'val_batch_size': 32,
        'test_batch_size': 32,

        # data augment
        'AutoAugment': False,
        'random_erasing': False,
        'RandAugment': False,
        'Mixup': False,
        'Cutmix': False,
        'Manifold_Mixup': False,
        'MoEx': False,                   # 注意：MoEx为True时，需要同步指定model_name
        'StyleMix': False,                # 注意：StyleMix为True时，需要同步指定StyleMix_method
        'StyleMix_method': 'StyleCutMix_Auto_Gamma',   # 'StyleMix', 'StyleCutMix', 'StyleCutMix_Auto_Gamma'

        # model
        'mode': 'train',
        'model_name': 'resnet18',
        #'model_name': 'moex_resnet18',
        'pretrained': False,    #True
        'gem': False,
        'feat_dims': 512,
        'id_nums': 2,

        # optimizer
        'max_epochs': 50,
        'weight_decay': 0.0005,
        'gamma': 0.1,
        'learning_rate': 3e-4,
        'lr_scheduler': 'step',   # step
        'step_size': 20,
        'warmup_iters': 5,
        'milestones': [15, 35],
        'cuda': True,

        # save path
        'log_dir': 'logs_platform',
        'snap': 'resnet50_256x256',
        'print_interval': 100,
        'valid_interval': 1 
}
