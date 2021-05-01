

def adapt_to_ex(model):


    self.conv_sections = conv_sections
    self.glob_av_pool = torch.nn.AdaptiveAvgPool2d(output_size=1)
    self.linear_sections = linear_sections
    self.head = head
    self.input_shape = input_shape
    self.n_classes = head.out_elements

    self.data_augment = DataAugmentation(n_classes=self.n_classes)
    self.loss_func_CE_soft = CrossEntropyWithProbs().to(device)
    self.loss_func_CE_hard = torch.nn.CrossEntropyLoss().to(device)
    self.loss_func_MSE = torch.nn.MSELoss().to(device)
    self.creation_time = datetime.now()

    config = {}

    config['n_conv_l'] = len(model.conv_sections)
    config['n_conv_0'] = model.conv_sections[0].out_elements
    config['n_conv_1'] = model.conv_sections[1].out_elements if len(model.conv_sections) > 1 else 16
    config['n_conv_2'] = model.conv_sections[2].out_elements if len(model.conv_sections) > 2 else 16

    # Dense
    config['n_fc_l'] = len(model.linear_sections)
    config['n_fc_0'] = model.linear_sections[0].out_elements
    config['n_fc_1'] = model.linear_sections[1].out_elements if len(model.linear_sections) > 1 else 16
    config['n_fc_2'] = model.linear_sections[2].out_elements if len(model.linear_sections) > 2 else 16

    # Kernel Size
    config['kernel_size'] = model.linear_sections[0].kernel_size

    # Learning Rate
    lr = RangeParameter('lr_init', ParameterType.FLOAT, 0.00001, 1.0, True)

    # Use Batch Normalization
    bn = ChoiceParameter('batch_norm', ParameterType.BOOL, values=[True, False])

    # Batch size
    bs = RangeParameter('batch_size', ParameterType.INT, 1, 512, True)

    # Global Avg Pooling
    ga = ChoiceParameter('global_avg_pooling', ParameterType.BOOL, values=[True, False])

    b = FixedParameter('budget', ParameterType.INT, 25)

    i = FixedParameter('id', ParameterType.STRING, 'dummy')
